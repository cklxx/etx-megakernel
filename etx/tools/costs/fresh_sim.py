"""CPU-only, conditional cost estimate of the frozen fresh-address experiment.

All times are microseconds internally, as in etx.sim.protocol and etx.sim.xcd.
This is a deterministic representative handoff (a predicted p50, no tail model).
It is NOT a measurement. Staleness is a hardware property this model cannot decide.

Assumptions (each is unmeasured for this kernel):
* Fresh cached first loads use the measured L2 (same) / MALL (cross) latency.
  Subsequent cached trips use L1 idle or L2 loaded; bypass uses the 0.55 us
  coherent activation benchmark as a proxy, NOT a measured single-dword latency.
* Equal bandwidth shares among streaming CUs model synthetic contention. Helpers
  keep streaming during waits; idle helper polling traffic is not quantified.
* Stores/flags use a MALL-hop proxy. Cross publish uses the isolated fence cost;
  its loaded multiplier is the MALL loaded/idle ratio. Local drain is included
  in the store hop. This does not reuse evx's incompatible full-handoff numbers.
* Four waves overlap perfectly; each 256-word trip waits before the next trip.
  Seven readers run concurrently, sharing bandwidth; writer ACK polls serialize.
* Unknown check, barrier, timer, wake, observation-store, setup and host-I/O costs
  have zero additive contribution in this scenario. Zero is an omission, not a
  claim that the hardware operation is free. Wall estimates are conditional;
  actual budget fit and watchdog completion remain unknown.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from etx.tools.costs.fresh_handoff import matrix

ROOT = Path(__file__).resolve().parents[3]


def source(path: str, text: str) -> str:
    """Resolve an exact repository line; fail closed if evidence disappears."""
    for line, value in enumerate((ROOT / path).read_text().splitlines(), 1):
        if text in value:
            return f'{path}:{line}'
    raise ValueError(f'missing source: {path}: {text}')


@dataclass(frozen=True)
class Constant:
    value: float
    source: str
    status: str
    note: str


def check_constants(constants: dict[str, Constant]) -> None:
    """Reject unnamed/unsourced hardware inputs, including assumed inputs."""
    for name, c in constants.items():
        if not name or not c.source or not c.note or c.status not in {'measured', 'derived', 'assumed'}:
            raise ValueError(f'unsourced named constant: {name}')
        path, sep, line = c.source.rpartition(':')
        if not sep or not line.isdigit() or not (ROOT / path).is_file():
            raise ValueError(f'invalid source for {name}: {c.source}')
        if not 1 <= int(line) <= len((ROOT / path).read_text().splitlines()):
            raise ValueError(f'invalid source line for {name}')
        if not math.isfinite(c.value) or c.value < 0:
            raise ValueError(f'invalid value for {name}')


def constants() -> dict[str, Constant]:
    out = {}

    def add(name, value, status, path, needle, note):
        out[name] = Constant(value, source(path, needle), status, note)

    model = 'docs/qwen/MODEL-vllm-kernels.md'
    kernel = 'bench/sync/fresh_handoff.h'
    for name, value, needle in (
        ('l1_idle_us', .103, '| vector round trip L1 /'),
        ('l2_idle_us', .143, '| vector round trip L1 /'),
        ('mall_idle_us', .261, '| vector round trip L1 /'),
        ('l2_loaded_us', .73, '| the same, loaded'),
        ('mall_loaded_us', .71, '| the same, loaded'),
        ('inv_idle_us', .19, '| buffer_inv sc1 alone'),
        ('inv_loaded_us', .63, '| buffer_inv sc1 alone'),
        ('launch_us', 1.61, '| kernel boundary, empty / + first load | 1.5'),
    ):
        add(name, value, 'measured', model, needle, 'MI300X 2026-09-28b; different microbenchmark, not fresh-handoff validation')
    add('bypass_proxy_us', .55, 'assumed', 'docs/qwen/VALIDATION-baremetal-2026-10-01.md',
        '| activation read: coherent', 'Whole activation read measured 0.55 us on bare metal; single-dword transfer is unknown')
    add('fence_us', .181, 'measured', 'etx/machine/arch/gfx942.yaml',
        't_fence_ns: 181', 'Isolated buffer_wbl2 sc1 plus completion wait, ns / 1000')
    add('bw_bytes_us', 4.3e6, 'assumed', 'AGENTS.md', 'about 4.3 TB/s',
        'Repository achievable ceiling, TB/s converted to bytes/us; not a per-CU streaming measurement')
    add('cus', 304, 'derived', 'etx/machine/arch/gfx942.yaml', 'count: 8, cus: 38', '8 XCDs times 38 active CUs')
    for name, value, needle, note in (
        ('threads', 256, 'constexpr int T', 'Threads per participating block'),
        ('word_bytes', 4, 'uint32_t* payload', 'uint32_t size in bytes'),
        ('line_bytes', 128, 'struct alignas(128)', 'Slot and flag alignment'),
        ('busy_region_bytes', 2097152, 'BUSY_WORDS =', '512 * 1024 uint32_t words per CU'),
        ('fanout_readers', 7, 'const int readers', 'One reader on each other XCD'),
        ('epochs', 2, 'epoch<=2', 'Separate launches per case'),
        ('wait_loads', 2, 'return !flag(&c->abort)', 'Successful wait reads target and abort; failed polls are unknown'),
    ):
        add(name, value, 'derived', kernel, needle, note)
    add('budget_s', 28800, 'derived', 'examples/sessions/fresh_handoff.py',
        '--budget-seconds 28800', 'Frozen matrix runner budget, seconds')
    # Explicit scenario choices, not fabricated hardware measurements.
    for name, value, needle, note in (
        ('share_weight', 1, '* Equal bandwidth', 'Equal per-CU stream weights; achieved helper bandwidth unknown'),
        ('check_us_per_word', 0, '* Unknown check', 'Omitted pattern arithmetic, compare and mismatch atomics per word'),
        ('extra_flow_us', 0, '* Unknown check', 'Omitted barriers, timers, polling sleep/phase, instrumentation and loop overhead per handoff'),
        ('setup_us', 0, '* Unknown check', 'Omitted placement, rendezvous and cooperative launch excess per launch'),
        ('host_s_per_case', 0, '* Unknown check', 'Omitted allocation/memsets, copies, CSV, process and host wait overhead per case'),
    ):
        add(name, value, 'assumed', 'etx/tools/costs/fresh_sim.py', needle, note)
    check_constants(out)
    return out


def simulate_case(case: dict, named: dict[str, Constant] | None = None) -> dict:
    """One representative slot; repeated algebraically, never allocate GPU payloads."""
    named = constants() if named is None else named
    check_constants(named)
    p = {k: c.value for k, c in named.items()}
    size, rounds = case['bytes'], case['handoffs']
    readers = int(p['fanout_readers']) if case['topology'] == 'fanout' else 1
    negative = case['mode'] in ('forced_old', 'primed_stale')
    loaded = case['load'] == 'synthetic_busy'
    cross = case['topology'] != 'same'
    helpers = int(p['cus']) - readers - 1 if loaded else 0
    bw = p['bw_bytes_us']
    store_rate = bw / (1 + helpers * p['share_weight'])
    read_rate = bw / (readers + helpers * p['share_weight'])
    mall = p['mall_loaded_us'] if loaded else p['mall_idle_us']
    cached_first = p['l2_loaded_us'] if loaded else p['l2_idle_us']
    if cross:
        cached_first = mall
    cached_trip = p['l2_loaded_us'] if loaded else p['l1_idle_us']
    bypass = max(p['bypass_proxy_us'], mall) if loaded else p['bypass_proxy_us']
    first = bypass if case['mode'] == 'bypass' else cached_first
    trip = bypass if case['mode'] == 'bypass' else cached_trip
    words = size // int(p['word_bytes'])
    trips = math.ceil(words / p['threads'])
    # Thread 0's first word is reused in bulk, but other lanes still issue trip 0.
    bulk = max(trips * trip, (size - p['word_bytes']) / read_rate)
    store = max(mall, size / store_rate)
    publish = p['fence_us'] * mall / p['mall_idle_us'] if cross else 0
    stages = [
        ('writer_store', store),
        ('writeback_wait' if cross else 'local_waitcnt', publish),
        ('flag_publish', mall),
        ('reader_poll', p['wait_loads'] * mall),
        ('first_payload_load', max(first, p['word_bytes'] / read_rate)),
        ('bulk_payload_load', bulk),
        ('per_word_check', math.ceil(words / p['threads']) * p['check_us_per_word']),
        ('ack_publish', mall),
        ('writer_ack_poll', readers * p['wait_loads'] * mall),
        ('unmeasured_flow_overhead', p['extra_flow_us']),
    ]
    events, now = [], 0.0
    for stage, duration in stages:
        events.append(dict(stage=stage, start_us=now, end_us=now + duration))
        now += duration
    # Both controls prime an old pattern before the timed writer round trip.
    prime = (store + publish + mall + p['wait_loads'] * mall
             + max(trips * cached_trip, size / read_rate)
             + mall + readers * p['wait_loads'] * mall) if negative else 0
    stride = math.ceil(size / p['line_bytes']) * int(p['line_bytes'])
    written = size * rounds * (2 if negative else 1)
    read = written * readers
    launch_s = (p['launch_us'] + p['inv_idle_us'] + p['setup_us'] + rounds * (now + prime)) / 1e6
    return dict(case=dict(case), readers=readers, helpers=helpers,
                busy_region_bytes=p['busy_region_bytes'], busy_working_set_bytes=helpers * p['busy_region_bytes'],
                slot_stride_bytes=stride, slots=1 if negative else rounds,
                payload_allocation_bytes=stride * (1 if negative else rounds),
                timed_written_bytes_per_launch=size * rounds, timed_read_bytes_per_launch=size * rounds * readers,
                written_bytes_per_launch=written, read_bytes_per_launch=read,
                bandwidth_floor_s_per_launch=(written + read) / bw / 1e6,
                predicted_p50_full_flow_us=now, reader_first_load_us=events[4]['end_us'] - events[4]['start_us'],
                prime_us_per_handoff=prime, stages=events,
                predicted_wall_s_per_launch=launch_s,
                predicted_wall_s_per_case=p['epochs'] * launch_s + p['host_s_per_case'],
                conditional_fits_budget=p['epochs'] * launch_s + p['host_s_per_case'] <= p['budget_s'])


def report() -> dict:
    named = constants()
    cases = [simulate_case(c, named) for c in matrix()['cases']]
    for row in cases:
        if row['case']['mode'] in ('cached', 'bypass'):
            other = simulate_case({**row['case'], 'mode': 'bypass' if row['case']['mode'] == 'cached' else 'cached'}, named)
            sign = 1 if row['case']['mode'] == 'cached' else -1
            row['paired_bypass_minus_cached_us'] = sign * (other['predicted_p50_full_flow_us'] - row['predicted_p50_full_flow_us'])
    epochs = named['epochs'].value
    positive = [r for r in cases if r['case']['mode'] in ('cached', 'bypass')]
    written = epochs * sum(r['written_bytes_per_launch'] for r in positive)
    read = epochs * sum(r['read_bytes_per_launch'] for r in positive)
    total = sum(r['predicted_wall_s_per_case'] for r in cases)
    timed_written = epochs * sum(r['timed_written_bytes_per_launch'] for r in cases)
    timed_read = epochs * sum(r['timed_read_bytes_per_launch'] for r in cases)
    unknowns = [dict(name=k, **asdict(c)) for k, c in named.items() if c.status == 'assumed']
    for name, note in (
        ('fresh_cache_latency', 'L2/MALL first-load and L1 bulk mapping, including negative-control reuse, is unvalidated.'),
        ('store_and_flag_latency', 'MALL-hop proxy for stores and coherent flags; failed poll count/backoff and idle helper traffic unknown.'),
        ('loaded_writeback', 'Idle fence scaled by loaded/idle MALL ratio; payload-size dependence unknown.'),
        ('fanout_contention', 'Equal bandwidth sharing and perfectly concurrent readers; cache replication and wave overlap unknown.'),
        ('staleness', 'Hardware property; cannot infer stale rate or negative-control outcome from ordered model events.'),
        ('tails', 'No distribution measured; p50 is a deterministic scenario, p95/p99 and timeout probability unknown.'),
    ):
        unknowns.append(dict(name=name, value=None, status='assumed',
                             source=source('etx/tools/costs/fresh_sim.py', 'Assumptions (each'), note=note))
    return dict(evidence='CPU simulation only; predictions, not measurements. Staleness is a hardware property this model cannot decide.',
                method=__doc__, constants={k: asdict(v) for k, v in named.items()}, unknowns=unknowns,
                cases=cases, case_count=len(cases), positive_written_bytes=written, positive_read_bytes=read,
                timed_matrix_written_bytes=timed_written, timed_matrix_read_bytes=timed_read,
                matrix_written_bytes=epochs * sum(r['written_bytes_per_launch'] for r in cases),
                matrix_read_bytes=epochs * sum(r['read_bytes_per_launch'] for r in cases),
                timed_matrix_bandwidth_floor_s=(timed_written + timed_read) / named['bw_bytes_us'].value / 1e6,
                positive_bandwidth_floor_s=(written + read) / named['bw_bytes_us'].value / 1e6,
                matrix_bandwidth_floor_s=epochs * sum(r['bandwidth_floor_s_per_launch'] for r in cases),
                predicted_matrix_wall_s=total, budget_s=named['budget_s'].value,
                conditional_fits_budget=total <= named['budget_s'].value, actual_budget_fit='unknown',
                caveats=[
                    'The timed pass of all 144 cases moves 28.3 TB (7.08 written + 21.25 read); the positive cases alone are 21.25 TB, and the controls\' priming pass brings the matrix to 35.42 TB. Helper/host traffic is additional.',
                    'Bandwidth is not the wall-time driver: 28.3 TB / 4.3 TB/s is about 6.6 s, against the predicted 6598 s of handoff round trips.',
                    'Prologue invalidation occurs before busy streaming: idle 0.19 us once per launch, never per handoff; concurrent CU invalidation/rendezvous cost unknown.',
                    'evx 2.20/3.91 us vs trace 0.3-0.5/1.5-2.1 us (MODEL-vllm-kernels section 10.1) are different protocols, not component costs.',
                    'Section 19 phase floors and October 1 decode wall overhead do not calibrate this probe.',
                    'Budget fit is conditional on zero omitted overhead; no correctness, promotion or hardware timing gate is evaluated.',
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, metavar='OUT')
    args = parser.parse_args()
    result = report()
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + '\n')
    print(result['evidence'])
    for r in result['cases']:
        print(f"{r['case']['id']}: p50 flow {r['predicted_p50_full_flow_us']:.3f} us; "
              f"first {r['reader_first_load_us']:.3f} us; floor {r['bandwidth_floor_s_per_launch']:.6f} s; "
              f"launch wall {r['predicted_wall_s_per_launch']:.3f} s")
    print(f"Matrix: {result['case_count']} cases; conditional wall {result['predicted_matrix_wall_s']:.3f} s / "
          f"{result['budget_s']:.0f} s; fits={result['conditional_fits_budget']}; actual fit unknown.")
    print(f"Positive payload bandwidth floor: {result['positive_bandwidth_floor_s']:.3f} s.")
    for caveat in result['caveats']:
        print(caveat)
    print('Unknowns: ' + ', '.join(u['name'] for u in result['unknowns']))


if __name__ == '__main__':
    main()
