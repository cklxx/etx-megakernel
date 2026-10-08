"""CPU-only hardware floors for one Qwen3-30B-A3B decode step on MI300X. This is NOT a measurement.

No phase floor comes from a measured task body. Each phase has one source tag:
* 'ISA hops x RTT': the exact dependent global-memory round trips of the phase, counted by
  etx/tools/isa_hops.py in the gfx942 ISA of the round11 build (clang 52226beb, ROCm 7.14.0a20260612,
  the build that matches the fingerprint). Each hop class has its own price: coherent loads and
  coherent store drains take the coherent round trip (RTT). Plain loads take the cached price,
  except after an acquire (`buffer_inv sc1`), where they take the RTT. Scalar loads, scratch
  reloads and plain store drains take the cached price.
* 'probe': the hw_stream achievable time of the phase (BUDGET achievable column).
* 'probe + ISA hops x RTT': the probe, plus the ISA hops before the first weight-stream load,
  plus the store drain.
* 'bytes/BW': only the byte floor applies (lm_head).
* 'unknown': no hop count and no probe. The phase adds only its bytes / bandwidth.

serial floor  = 48 x (sum over phases of max(bytes/BW, phase floor) + handoffs)
                + sum over tail phases of max(bytes/BW, tail floor)
overlap floor = max(step bytes/BW, 48 x (sum over phases of ISA chain + handoffs) + tail ISA chains)
The overlap floor assumes that every byte streams under earlier phases, with unlimited capacity.
The handoff term is the measured 9 edges per layer. It is a measurement of the edge, not of a body.

Inputs: docs/FLOORS-2026-10-08.md holds the round11 phase table, recomputed from the raw trace,
and the ISA hop table with the MATCH .s lines. `round11_from_trace` recomputes the phase table.
`python -m etx.tools.isa_phases` recomputes the ISA hop table.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
DOC = 'docs/FLOORS-2026-10-08.md'
BUDGET = 'docs/BUDGET-2026-10-01.md'
TILES = 'docs/TILES-2026-10-07.md'
VALID = 'docs/VALIDATION-baremetal-2026-10-01.md'
BW_DOC = 'docs/BANDWIDTH-2026-10-07.md'
PHASES = ('qkv', 'prep', 'attn', 'reduce', 'o_proj', 'router', 'topk', 'moe_up', 'moe_down')
TAIL = ('prologue', 'in_norm', 'lm_head', 'argmax')
ISA_TAIL = ('prologue', 'in_norm', 'argmax')            # tail phases with an ISA hop row
TAGS = ('ISA hops x RTT', 'probe', 'probe + ISA hops x RTT', 'bytes/BW', 'unknown')
LAYERS = 48
BANDWIDTHS_TBS = (4.3, 3.5)
ISA_FILE = 'isa_r11/MATCH/megakernel_d0.gfx942.s'      # under /workspace on the shared box
ISA_COMPILER = 'AMD clang 23.0.0git 52226beb (TheRock ROCm 7.14.0a20260612), tile ae1529c'
ISA_FINGERPRINT = {'vgpr': 128, 'scratch_bytes': 52, 'vgpr_spill': 14, 'lds_bytes': 41448}
CONTROL_COMPILER = 'ROCm 7.2.4 clang 22.0.0git f58b06dc (no fingerprint match: spill 9, scratch 32)'
WG_MEDIAN_RTT_US = 0.28                                # hw_cohload.log median of one workgroup


def source(path: str, text: str, after: str | None = None) -> tuple[str, str]:
    """Return (location, line) for an exact repository line; fail closed if it disappears.

    With `after`, the search starts below the first line that contains `after`.
    """
    seen = after is None
    for line, value in enumerate((ROOT / path).read_text().splitlines(), 1):
        if not seen:
            seen = after in value
        elif text in value:
            return f'{path}:{line}', value
    raise ValueError(f'missing source: {path}: {text}')


def cells(line: str) -> list[str]:
    return [c.strip().strip('*').strip('`') for c in line.strip().strip('|').split('|')]


def _num(path_line: tuple[str, str], pattern: str) -> list[float]:
    where, line = path_line
    found = re.findall(pattern, line)
    if not found:
        raise ValueError(f'no number at {where}')
    return [float(x) for x in (found[0] if isinstance(found[0], tuple) else found)]


# ------------------------------------------------------------------ sourced constants
def rtt_range_us() -> tuple[tuple[float, float], str]:
    src = source(TILES, 'A coherent load that misses L2 takes about')
    lo, hi = _num(src, r'(\d+\.\d+) to (\d+\.\d+) us')
    return (lo, hi), src[0]


def cached_hop_us() -> tuple[float, str]:
    src = source(VALID, 'ns without it')
    ns = _num(src, r'against (\d+\.\d+) ns without it')[0]
    return ns / 1000.0, src[0]


def handoff_us_per_layer() -> tuple[float, str]:
    where, line = source(BUDGET, '| handoff, 9 edges at 1.00 us |')
    return float(cells(line)[1]), where


def measured_span_us() -> tuple[float, str]:
    where, line = source(BW_DOC, 'us device span.')
    found = re.findall(r'([\d,]+\.\d+) us device span', line)
    if not found:
        raise ValueError(f'no span at {where}')
    return float(found[0].replace(',', '')), where


def probes_us() -> dict[str, tuple[float | None, str]]:
    """BUDGET achievable column: bytes per task / hw_stream per-workgroup rate. 'latency' = no probe."""
    out = {}
    for name in PHASES:
        where, line = source(BUDGET, f'| {name} | ', after='## The table (one layer')
        row = cells(line)
        if len(row) != 10:
            raise ValueError(f'unexpected BUDGET row: {where}')
        out[name] = (None if row[7] == 'latency' else float(row[7]), where)
    return out


# Static evidence of the hop chains: (revision, path, line, text). The ETX lines are in this repository at the
# round11 commit. The vLLM lines are in a vLLM v0.27.1 checkout. tests/test_floors.py checks each line where the
# revision is present. See docs/FLOORS-2026-10-08.md for the chains.
TILE_REV = 'ae1529c'
VLLM_REV = '6e448d0ea9bf3d88d898b65449ca6dc2aec170ac'
TILE_EVIDENCE = (
    ('etx/ops/attnprep.py', 50, 'float e0 = dsl_a_ldbf(row + 2 * lane)'),
    ('etx/ops/attnprep.py', 52, 'ss = etx_wave_sum(ss);'),
    ('etx/ops/attnprep.py', 57, 'dsl_a_ldll_u(positions)'),
    ('etx/ops/attnprep.py', 63, '__bfloat162float(cosp[d0])'),
    ('etx/ops/attnprep.py', 68, 'dsl_a_stbf(row + 2 * lane, o0)'),
    ('etx/ops/attnprep.py', 70, 'dsl_a_ldll_u(slot_mapping)'),
    ('etx/ops/attnprep.py', 73, 'dsl_a_stbf(kdst'),
    ('etx/ops/topk.py', 23, 'if (etx_wave() != 0) return;'),
    ('etx/ops/topk.py', 31, 'dsl_a_ldbf(logits + e)'),
    ('etx/ops/topk.py', 51, 'dsl_a_sti(ids + t, bi)'),
    ('etx/ops/topk.py', 54, 'dsl_a_stf(wts + t'),
    ('etx/ops/gemv.py', 197, 'dsl_a_ld64((const unsigned long long*)(x + 8 * i))'),
    ('etx/ops/gemv.py', 220, 'DslArgs dsl_args(const unsigned char* a)'),
    ('etx/ops/gemv.py', 236, 'acc += dsl_a_ldbf(d.rows + (size_t)s * K + i)'),
    ('etx/ops/gemv.py', 238, 'dsl_a_ldbf(d.resid_in + i)'),
    ('etx/ops/gemv.py', 242, 'ss = etx_block_sum(ss, red);'),
    ('etx/ops/gemv.py', 245, 'd.ln[i]'),
    ('etx/ops/gemv.py', 298, 'w[u] = base[(size_t)(sb + u) * ETX_WAVE_SIZE]'),
    ('etx/ops/gemv.py', 321, 'dsl_a_stbf(out + r'),
    ('etx/ops/gemv.py', 334, 'dsl_a_ldi(d.ids + s)'),
    ('etx/ops/gemv.py', 357, 'dsl_a_ldi(d.ids + s)'),
    ('etx/ops/gemv.py', 363, 'dsl_a_ldf(d.wts + s)'),
    ('etx/ops/gemv.py', 378, 'const int nblk'),
    ('etx/ops/coh.py', 27, '__hip_atomic_store'),
    ('etx/codegen/kernel.py', 320, 'ETX_RELEASE_DOMAIN();'),
    ('etx/runtime/include/etx/import.h', 108, 'etx_imp_sgpr64(a4[0])'),
)
VLLM_EVIDENCE = (
    ('csrc/rocm/attention.cu', 321, 'void paged_attention_ll4mi_QKV_mfma16_kernel('),
    ('csrc/rocm/attention.cu', 362, 'const int seq_len = seq_lens[seq_idx];'),
    ('csrc/rocm/attention.cu', 438, 'block_table_seq[kblock_idx]'),
    ('csrc/rocm/attention.cu', 551, 'block_table_seq[vblock_idx]'),
    ('csrc/rocm/attention.cu', 780, 'max_logits[offset] = partition_qk_max'),
    ('csrc/rocm/attention.cu', 906, '*out_ptr_B16x8 = vout[h]'),
    ('csrc/rocm/attention.cu', 1420, 'void paged_attention_ll4mi_reduce_kernel('),
    ('csrc/rocm/attention.cu', 1442, 'const int seq_len = seq_lens[seq_idx];'),
    ('csrc/rocm/attention.cu', 1469, 'max_logits_ptr[valid_partition[i]]'),
    ('csrc/rocm/attention.cu', 1489, 'exp_sums_ptr[valid_partition[i]]'),
    ('csrc/rocm/attention.cu', 1538, 'tmp_out_ptr[lastj_offset]'),
    ('csrc/rocm/attention.cu', 1619, 'out_ptr[threadIdx.x] = from_float<scalar_t>(acc)'),
)


# ------------------------------------------------------------------ the evidence document
@dataclass(frozen=True)
class Measured:
    name: str
    tasks: int
    gb_step: float
    crit_body: float
    mean_body: float
    delta: float
    start: float
    where: str


@dataclass(frozen=True)
class Hops:
    """Exact ISA hop counts of one phase. `pre_*`: the hops before the first weight-stream load."""
    name: str
    tag: str
    region: str
    preset: str
    coherent: int
    plain: int
    scalar: int
    scratch: int
    store_coherent: int
    store_plain: int
    pre_coherent: int
    pre_plain: int
    pre_scalar: int
    chain: str
    where: str

    def cost(self, rtt: float, cached: float) -> float:
        """Price of the whole chain, from the region entry to the publish."""
        plain = rtt if self.preset == 'acquired' else cached
        return (self.coherent * rtt + self.plain * plain + (self.scalar + self.scratch) * cached
                + self.store_coherent * rtt + self.store_plain * cached)

    def pre_cost(self, rtt: float, cached: float) -> float:
        """Price of the hops before the stream and of the store drain at the end."""
        plain = rtt if self.preset == 'acquired' else cached
        return (self.pre_coherent * rtt + self.pre_plain * plain + self.pre_scalar * cached
                + self.store_coherent * rtt + self.store_plain * cached)


def round11_table() -> dict[str, Measured]:
    out = {}
    for name in PHASES:
        where, line = source(DOC, f'| {name} | ', after='## Round11 phase table')
        r = cells(line)
        if len(r) != 7:
            raise ValueError(f'unexpected round11 row: {where}')
        out[name] = Measured(name, int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6]), where)
    return out


def tail_table() -> dict[str, tuple[float, float, str, str]]:
    """phase -> (GB per step, device us, tag, location)."""
    out = {}
    for name in TAIL:
        where, line = source(DOC, f'| {name} | ', after='## Tail table')
        r = cells(line)
        if len(r) != 4 or r[3] not in TAGS:
            raise ValueError(f'unexpected tail row: {where}')
        out[name] = (float(r[1]), float(r[2]), r[3], where)
    return out


def hop_table() -> dict[str, Hops]:
    """ISA hop table: the 9 layer phases and the tail phases with an ISA count."""
    out = {}
    for name in PHASES + ISA_TAIL:
        where, line = source(DOC, f'| {name} | ', after='## ISA hop table')
        r = cells(line)
        if len(r) != 15 or r[1] not in TAGS:
            raise ValueError(f'unexpected ISA hop row: {where}')
        n = [int(x) for x in r[4:13]]
        h = Hops(name, r[1], r[2], r[3], *n, r[13], f'{where}; {ISA_FILE}:{r[14]}')
        if h.preset not in ('warm', 'acquired'):
            raise ValueError(f'unknown preset: {where}')
        if min(n) < 0 or h.pre_coherent > h.coherent or h.pre_scalar > h.scalar:
            raise ValueError(f'hop counts out of order: {where}')
        out[name] = h
    return out


def wg_median_rtt_us() -> tuple[float, str]:
    src = source(DOC, 'median of one workgroup is')
    return _num(src, r'is (\d+\.\d+) us')[0], src[0]


# ------------------------------------------------------------------ arithmetic
def floors(bw_tbs: float = 4.3, rtt_us: float | None = None) -> dict:
    (rtt_lo, rtt_hi), rtt_src = rtt_range_us()
    rtt = rtt_lo if rtt_us is None else rtt_us
    t_cache, cache_src = cached_hop_us()
    hand, hand_src = handoff_us_per_layer()
    span, span_src = measured_span_us()
    probes = probes_us()
    meas, hop = round11_table(), hop_table()
    tail = tail_table()
    rows, flags = {}, []
    for p in PHASES:
        m, h = meas[p], hop[p]
        chain_us = h.cost(rtt, t_cache)
        probe = probes[p][0] if h.tag.startswith('probe') else None
        if h.tag.startswith('probe') and probe is None:
            raise ValueError(f'{p}: probe tag without a BUDGET probe')
        if h.tag == 'ISA hops x RTT':
            floor_lat = chain_us
        elif h.tag == 'probe + ISA hops x RTT':
            floor_lat = probe + h.pre_cost(rtt, t_cache)
        elif h.tag == 'probe':
            floor_lat = probe
        else:
            raise ValueError(f'{p}: tag {h.tag} is not valid for a layer phase')
        stream = m.gb_step / LAYERS * 1e3 / bw_tbs              # GB per step -> MB per layer -> us
        serial = max(stream, floor_lat)
        if floor_lat > m.crit_body:
            flags.append(p)
        rows[p] = {'tag': h.tag, 'region': h.region, 'preset': h.preset,
                   'hops': {'coherent': h.coherent, 'plain': h.plain, 'scalar': h.scalar, 'scratch': h.scratch,
                            'store_coherent': h.store_coherent, 'store_plain': h.store_plain},
                   'chain_us': chain_us, 'probe_us': probe, 'phase_floor_us': floor_lat, 'stream_us': stream,
                   'serial_us': serial, 'crit_body_us': m.crit_body, 'body_excess_us': m.crit_body - serial,
                   'chain': h.chain, 'evidence': h.where, 'measured': m.where,
                   'probe_source': probes[p][1] if probe is not None else None}
    tail_rows = {}
    for t, (gb, dev, tag, where) in tail.items():
        stream = gb * 1e3 / bw_tbs
        h = hop.get(t)
        if tag == 'ISA hops x RTT':
            if h is None:
                raise ValueError(f'{t}: ISA tag without an ISA hop row')
            fl = h.cost(rtt, t_cache)
        else:
            fl = 0.0
        if fl > dev:
            flags.append(t)
        tail_rows[t] = {'tag': tag, 'floor_us': fl, 'stream_us': stream, 'serial_us': max(stream, fl),
                        'device_us': dev, 'chain_us': fl, 'measured': where, 'evidence': h.where if h else None}
    tail_stream = sum(r['stream_us'] for r in tail_rows.values())
    stream_total = LAYERS * sum(r['stream_us'] for r in rows.values()) + tail_stream
    chain = LAYERS * (sum(r['chain_us'] for r in rows.values()) + hand) + sum(r['chain_us'] for r in tail_rows.values())
    serial = LAYERS * (sum(r['serial_us'] for r in rows.values()) + hand) + sum(r['serial_us'] for r in tail_rows.values())
    overlap = max(stream_total, chain)
    body_excess = LAYERS * sum(r['body_excess_us'] for r in rows.values())
    start = LAYERS * sum(m.start for m in meas.values())
    unknown = sorted([p for p, r in rows.items() if r['tag'] == 'unknown'] +
                     [p for p, r in tail_rows.items() if r['tag'] == 'unknown'])
    return {
        'bandwidth_tbs': bw_tbs,
        'rtt_us': rtt,
        'cached_hop_us': t_cache,
        'bandwidth_floor_us': stream_total,
        'hop_chain_us': chain,
        'overlap_floor_us': overlap,
        'overlap_bound_by': 'bandwidth' if stream_total >= chain else 'hop chain',
        'serial_floor_us': serial,
        'measured_span_us': span,
        'prefetch_gain_max_us': serial - overlap,
        'non_prefetch_gain_max_us': span - serial,
        'split_of_non_prefetch_gain_us': {
            'body_excess': body_excess,
            'start_delay': start,
            'rest': span - serial - body_excess - start,
        },
        'body_excess_by_phase_us': {k: LAYERS * v['body_excess_us'] for k, v in rows.items()},
        'utilization_ceiling': {'serial': stream_total / serial, 'overlap': stream_total / overlap},
        'unknown_phases': unknown,
        'lower_bound': True,
        'floor_above_measured_body': flags,
        'per_layer': rows,
        'tail': tail_rows,
        'isa': {'file': ISA_FILE, 'compiler': ISA_COMPILER, 'fingerprint': ISA_FINGERPRINT},
        'sources': {'rtt': rtt_src, 'rtt_range_us': [rtt_lo, rtt_hi], 'cached_hop': cache_src,
                    'handoff': hand_src, 'span': span_src, 'tail': {k: v[3] for k, v in tail.items()}},
    }


def grid() -> list[dict]:
    (lo, hi), _ = rtt_range_us()
    return [floors(bw, rtt) for rtt in (lo, hi) for bw in BANDWIDTHS_TBS]


def sensitivity() -> list[dict]:
    """The same floors at the one-workgroup median round trip of hw_cohload.log."""
    med, _ = wg_median_rtt_us()
    return [floors(bw, med) for bw in BANDWIDTHS_TBS]


# ------------------------------------------------------------------ round11 recomputation (needs the archive)
def round11_from_trace(plan_path: str | Path, raw_path: str | Path, first: int = 1, last: int = 45) -> dict[str, dict]:
    """Per-phase means over layers first..last from the raw trace: critical body, mean body, delta, start."""
    from etx.tools import timeline as tl
    tr = tl.build_trace(tl.load_plan(plan_path), tl.read_raw(raw_path))
    groups: dict[tuple[str, int], list] = {}
    for t in tr.tasks:
        m = re.match(r'^([a-z_]+?)_(\d+)(_x)?$', t.name)
        if m and m.group(1) in PHASES:
            groups.setdefault((m.group(1), int(m.group(2))), []).append(t)
    order = [(p, L) for L in range(LAYERS) for p in PHASES]
    first_ready = {k: min(t.ready for t in v) for k, v in groups.items()}
    out = {}
    for p in PHASES:
        crit, mean, delta, start = [], [], [], []
        for L in range(first, last + 1):
            v = groups[(p, L)]
            c = max(v, key=lambda t: t.done)
            crit.append(c.done - c.ready)
            mean.append(sum(t.done - t.ready for t in v) / len(v))
            nxt = order[order.index((p, L)) + 1]
            delta.append(first_ready[nxt] - first_ready[(p, L)])
            start.append(c.ready - first_ready[(p, L)])
        out[p] = {'tasks': len(groups[(p, first)]), 'crit_body': statistics.mean(crit),
                  'mean_body': statistics.mean(mean), 'delta': statistics.mean(delta),
                  'start': statistics.mean(start)}
    return out


def table(results: list[dict]) -> str:
    lines = ['RTT  BW   | bandwidth  chain  overlap  serial  measured | prefetch  non-prefetch | bound']
    for r in results:
        lines.append(f"{r['rtt_us']:.2f} {r['bandwidth_tbs']:.1f}  | {r['bandwidth_floor_us']:8.1f} "
                     f"{r['hop_chain_us']:7.1f} {r['overlap_floor_us']:8.1f} {r['serial_floor_us']:7.1f} "
                     f"{r['measured_span_us']:8.1f} | {r['prefetch_gain_max_us']:8.1f} {r['non_prefetch_gain_max_us']:12.1f}"
                     f" | {r['overlap_bound_by']}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--table', action='store_true', help='print a summary table instead of JSON')
    parser.add_argument('--trace', nargs=2, metavar=('PLAN', 'RAW'), help='recompute the round11 phase table')
    args = parser.parse_args(argv)
    if args.trace:
        print(json.dumps(round11_from_trace(*args.trace), indent=2))
        return 0
    results = grid() + sensitivity()
    print(table(results) if args.table else json.dumps(results, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
