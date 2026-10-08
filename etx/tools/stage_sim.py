"""CPU-only, conditional estimate of staging o_proj weights into LDS across the attention chain.

Lever (proposed, not built): while prep -> attn -> reduce run on a few workers, every o_proj
worker is idle. Before it polls its o_proj event, the worker issues `buffer_load ... lds`
(gfx942: 32 bit per lane, no VGPR destination) for the first part of its weight slice into the
LDS bytes that no tile uses. The o_proj body then reads that part from LDS and streams only the rest.

All times are microseconds. This is NOT a measurement. It answers one question before any GPU
time: is the predicted gain above the repository's KEEP threshold, and what blocks it.

Assumptions (each is unmeasured for this lever):
* The round11 trace (CU=256) describes the o_proj body; the shipped build (4.823 ms) is newer.
* The staged share of the body scales with staged bytes (scenario A) or with the measured
  per-workgroup stream rate at ~296 workgroups (scenario B). The low/high range spans both.
* Fixed per-task cost (arrive, acquire, publish) equals the measured empty router task.
* The 29,736 B group segment of the 2026-10-01 dsl build still holds; LDS above it is free.
* LDS read rate and engine clock are public peak values; the visibility wait before the
  first LDS read and any contention with attention's KV reads are omitted (zero is an
  omission, not a claim that they are free).
* A gain on o_proj's critical body shortens the step one for one (o_proj is serial).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAT = 'docs/LATENCY-2026-10-07.md'


def source(path: str, text: str) -> str:
    """Resolve an exact repository line; fail closed if the evidence disappears."""
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


def check_constants(named: dict[str, Constant]) -> None:
    for name, c in named.items():
        if not name or not c.note or c.status not in {'measured', 'derived', 'assumed'}:
            raise ValueError(f'unsourced named constant: {name}')
        path, sep, line = c.source.rpartition(':')
        if not sep or not line.isdigit() or not (ROOT / path).is_file():
            raise ValueError(f'invalid source for {name}: {c.source}')
        if not 1 <= int(line) <= len((ROOT / path).read_text().splitlines()):
            raise ValueError(f'invalid source line for {name}')
        if not math.isfinite(c.value) or c.value < 0:
            raise ValueError(f'invalid value for {name}')


def constants() -> dict[str, Constant]:
    out: dict[str, Constant] = {}

    def add(name, value, status, path, needle, note):
        out[name] = Constant(float(value), source(path, needle), status, note)

    trace = 'Round11 device trace, Qwen3-30B-A3B bf16, batch 1, one MI300X VF, 48 layers'
    add('layers', 48, 'derived', LAT, 'All rows aggregate 48 layers.', 'Decoder layers per token')
    add('oproj_bytes', 805306368, 'measured', LAT, '| o_proj | 805306368 |', 'Logical o_proj weight bytes per token; ' + trace)
    add('oproj_tasks', 12288, 'measured', LAT, '| o_proj | 12288 | 256 |', 'o_proj tasks per token (256 per layer); ' + trace)
    add('oproj_body_mean_us', 5.451, 'measured', LAT, '| o_proj | 12288 | 256 | 5.451', 'Mean o_proj task body; ' + trace)
    add('oproj_crit_body_us', 269.44, 'measured', LAT, '| o_proj | 287.47 |', 'o_proj critical-path body, all layers; ' + trace)
    add('oproj_crit_gap_us', 105.26, 'measured', LAT, '| o_proj | 287.47 |', 'o_proj critical dependency gap, all layers; staging leaves it unchanged')
    for phase, value, needle in (('prep', 287.28, '| prep | 305.34 |'), ('attn', 310.84, '| attn | 344.76 |'),
                                 ('reduce', 268.52, '| reduce | 310.12 |')):
        add(f'{phase}_crit_body_us', value, 'measured', LAT, needle, f'{phase} critical-path body, all layers; part of the idle window')
    add('empty_task_us', .506, 'measured', LAT, 'The 6,144 empty tasks average 0.506 us.',
        'Body of a task with no rows (arrive, acquire, exit): fixed per-task cost proxy')
    add('lds_total_bytes', 65536, 'derived', 'etx/machine/arch/gfx942.yaml', 'lds_kb: 64', 'LDS per CU')
    add('lds_used_bytes', 29736, 'measured', 'docs/VALIDATION-baremetal-2026-10-01.md', 'LDS is 29736, not 41448',
        'Group segment of the dsl control build (CU=256, import prune), VF and bare metal; 41448 is the all-imported build')
    add('threads', 1024, 'derived', 'examples/vllm_llm/model.py', 'THREADS = 1024', 'Megakernel workgroup size')
    add('lds_dma_bytes_per_lane', 4, 'derived', 'etx/machine/arch/gfx942.yaml', 'async_copy_to_lds: dword_only',
        'buffer_load ... lds moves 32 bit per lane and needs no VGPR destination')
    add('vgpr_per_wave', 128, 'measured', 'docs/BUDGET-2026-10-01.md', 'VGPR: 512/SIMD ÷ 128/wave = 4',
        'Megakernel VGPRs per wave (15 spills already)')
    add('regs_per_lane', 512, 'derived', 'etx/machine/arch/gfx942.yaml', 'regs_per_lane: 512', 'Register file entries per SIMD lane')
    add('simds_per_cu', 4, 'derived', 'etx/machine/arch/gfx942.yaml', 'simds_per_cu: 4', 'SIMDs per CU')
    add('wave_size', 64, 'derived', 'etx/machine/arch/gfx942.yaml', 'wave_size: 64', 'Lanes per wave')
    add('wg_rate_kb_us', 8.9, 'measured', 'examples/vllm_llm/model.py', '8.9 KB/us at 296',
        'hw_bench stream per-workgroup rate at 296 workgroups (gate_up shape, 2026-10-01)')
    add('ref_ms', 4.823, 'measured', 'README.md', '**ETX megakernel** | **4.823**', 'Shipped ETX ms/token (wall), 2026-10-02')
    add('spread_ms', .028, 'measured', 'docs/TILES-2026-10-07.md', 'same-binary spread of five fused runs is 0.028',
        'Same-binary spread of five fused runs')
    add('keep_ms', .030, 'derived', 'docs/TILES-2026-10-07.md', 'The mean gain is at or below -0.030 ms.',
        'Repository KEEP threshold for a flag A/B, ms per token')
    # Public peak values, not repository measurements.
    add('lds_bytes_per_clk', 128, 'assumed', 'etx/tools/stage_sim.py', '* LDS read rate and engine clock',
        'CDNA3 LDS peak bytes per clock per CU (public)')
    add('clock_mhz', 2100, 'assumed', 'etx/tools/stage_sim.py', '* LDS read rate and engine clock',
        'MI300X peak engine clock (public)')
    add('visibility_wait_us', 0, 'assumed', 'etx/tools/stage_sim.py', '* LDS read rate and engine clock',
        'Omitted s_waitcnt + barrier before the first LDS read of staged bytes')
    check_constants(out)
    return out


def simulate(named: dict[str, Constant] | None = None, path: str = 'lds') -> dict:
    """One o_proj task per worker; the critical task decides. path: 'lds' (buffer_load lds) or 'vgpr'."""
    named = constants() if named is None else named
    check_constants(named)
    p = {k: c.value for k, c in named.items()}
    layers = p['layers']
    tasks_per_layer = p['oproj_tasks'] / layers
    slice_bytes = p['oproj_bytes'] / p['oproj_tasks']
    step = p['threads'] * p['lds_dma_bytes_per_lane']          # bytes per workgroup-wide LDS DMA instruction
    free = max(0.0, p['lds_total_bytes'] - p['lds_used_bytes'])
    staged = min(slice_bytes, math.floor(free / step) * step)
    share = staged / slice_bytes
    extra_vgpr = math.ceil(staged / (p['threads'] * 4)) if path == 'vgpr' else 0
    waves_per_simd = p['threads'] / p['wave_size'] / p['simds_per_cu']
    vgpr_limit = p['regs_per_lane'] / waves_per_simd                # one 1024-thread workgroup must fit per CU
    stage_time = staged / (p['wg_rate_kb_us'] * 1000)
    window = (p['prep_crit_body_us'] + p['attn_crit_body_us'] + p['reduce_crit_body_us']) / layers
    feasible = staged > 0 and p['vgpr_per_wave'] + extra_vgpr <= vgpr_limit and stage_time <= window
    crit_body = p['oproj_crit_body_us'] / layers
    stream = max(0.0, crit_body - p['empty_task_us'])           # streaming share of the critical body
    hidden_a = share * stream
    hidden_b = min(stream, staged / (p['wg_rate_kb_us'] * 1000))
    lds_read = staged / (p['lds_bytes_per_clk'] * p['clock_mhz'])  # bytes / (bytes/clk * clk/us)
    cost = lds_read + p['visibility_wait_us']
    net_low = max(0.0, min(hidden_a, hidden_b) - cost) if feasible else 0.0
    net_high = max(0.0, max(hidden_a, hidden_b) - cost) if feasible else 0.0
    gain_low_ms, gain_high_ms = net_low * layers / 1000, net_high * layers / 1000
    verdict = ('INFEASIBLE' if not feasible else
               'BELOW_THRESHOLD' if gain_high_ms < p['keep_ms'] else
               'MARGINAL' if gain_low_ms < p['keep_ms'] else 'ABOVE_THRESHOLD')
    return dict(path=path, tasks_per_layer=tasks_per_layer, slice_bytes=slice_bytes, free_lds_bytes=free,
                staged_bytes=staged, staged_share=share, dma_instructions_per_lane=staged / step,
                extra_vgpr_per_lane=extra_vgpr, vgpr_limit=vgpr_limit, feasible=feasible,
                crit_body_us_per_layer=crit_body, stream_us_per_layer=stream,
                hidden_us_scenario_a=hidden_a, hidden_us_scenario_b=hidden_b, added_cost_us=cost,
                window_us_per_layer=window, stage_time_us=stage_time, fits_window=stage_time <= window,
                net_us_per_layer=(net_low, net_high), gain_ms_per_token=(gain_low_ms, gain_high_ms),
                gain_pct_of_ref=(100 * gain_low_ms / p['ref_ms'], 100 * gain_high_ms / p['ref_ms']),
                above_spread=gain_low_ms > p['spread_ms'], verdict=verdict)


def sweep_lds_used(named: dict[str, Constant] | None = None, used=(29736, 33792, 41448, 49152, 57344, 61440)) -> list[dict]:
    """Gain against the group segment of the current build (unverified locally)."""
    named = constants() if named is None else named
    rows = []
    for u in used:
        n = dict(named)
        n['lds_used_bytes'] = Constant(float(u), named['lds_used_bytes'].source, 'assumed', 'Sweep value')
        r = simulate(n)
        rows.append(dict(lds_used_bytes=u, staged_bytes=r['staged_bytes'], gain_ms_per_token=r['gain_ms_per_token'],
                         verdict=r['verdict']))
    return rows


def report() -> dict:
    named = constants()
    unknowns = [dict(name=k, **asdict(c)) for k, c in named.items() if c.status == 'assumed']
    for name, note in (
        ('current_group_segment', 'Read .amdhsa_group_segment_fixed_size of the current mk.s (rocm_build_check.sh); local, no GPU.'),
        ('lds_dma_rate', 'buffer_load ... lds rate per workgroup with 256 workgroups and a co-running attention phase.'),
        ('body_split', 'How much of the o_proj body is weight streaming versus activation wait, reduction and publish.'),
        ('kv_interference', 'Staging loads during attn may evict MALL-resident KV lines; use non-temporal loads and measure.'),
        ('critical_path', 'Whether a shorter o_proj body moves router\'s first ready time one for one.'),
    ):
        unknowns.append(dict(name=name, value=None, status='assumed',
                             source=source('etx/tools/stage_sim.py', 'Assumptions (each'), note=note))
    return dict(evidence='CPU simulation only; predictions, not measurements.', method=__doc__,
                constants={k: asdict(v) for k, v in named.items()},
                lds=simulate(named, 'lds'), vgpr=simulate(named, 'vgpr'), sweep=sweep_lds_used(named),
                unknowns=unknowns,
                prior=[
                    'fleet-mi300x docs/STATUS.md:30 (85b087e): prefetch of the next task\'s rows across the event wait into VGPRs, 8 x 16 B per thread, measured 4% slower (3.812 vs 3.673 ms) from register pressure. The vgpr path here reproduces that blocker.',
                    'fleet-mi300x docs/STATUS.md:72: idle-worker prefetch into the Infinity Cache, no gain. This lever targets LDS, not a cache level.',
                    'docs/TILES-2026-10-07.md: L2 prefetch is a measured loss on this path; not repeated here.',
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, metavar='OUT')
    args = parser.parse_args()
    r = report()
    if args.json:
        args.json.write_text(json.dumps(r, indent=2) + '\n')
    print(r['evidence'])
    for key in ('lds', 'vgpr'):
        s = r[key]
        lo, hi = s['gain_ms_per_token']
        print(f"{key}: staged {s['staged_bytes']:.0f}/{s['slice_bytes']:.0f} B ({100 * s['staged_share']:.1f}%), "
              f"+{s['extra_vgpr_per_lane']} VGPR, feasible={s['feasible']}; gain {lo:.3f}-{hi:.3f} ms/token "
              f"({s['gain_pct_of_ref'][0]:.2f}-{s['gain_pct_of_ref'][1]:.2f}%); stage {s['stage_time_us']:.2f} us "
              f"in a {s['window_us_per_layer']:.2f} us window; verdict {s['verdict']}")
    print('Sweep over the current group segment (B used -> staged B, gain ms, verdict):')
    for row in r['sweep']:
        lo, hi = row['gain_ms_per_token']
        print(f"  {row['lds_used_bytes']} -> {row['staged_bytes']:.0f}, {lo:.3f}-{hi:.3f}, {row['verdict']}")
    for line in r['prior']:
        print('Prior: ' + line)
    print('Unknowns: ' + ', '.join(u['name'] for u in r['unknowns']))


if __name__ == '__main__':
    main()
