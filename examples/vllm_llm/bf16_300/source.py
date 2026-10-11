"""Isolated source generation. Nothing in the production emitter/environment changes.

Probe pairs are ISSUE stamps (s_memrealtime, 10 ns/tick); only the retained
publication fence provides retirement. No extra fence is inserted at a stamp.
Per-pair sampling bounds state to two timestamps. FLUSH samples one named row,
not the sum of all flushes or a proof of which wave is on the critical path.
"""
import re
from etx.ops import gemv, prims, coh
from .protocol import require

# Each pair is its own build. Overlapping pairs must never be added together.
PAIRS = {
    "arguments": ("body_begin", "args"),
    "routing": ("args", "routing"),
    "stage": ("routing", "stage"),
    "zero": ("zero_begin", "zero_end"),
    "pipeline": ("pipeline_begin", "pipeline_end"),
    "flush": ("flush_begin", "flush_end"),
    "join": ("pipeline_end", "join"),
    "gather": ("join", "gather"),
    "reuse": ("gather", "reuse"),
    "publication": ("publish_begin", "publish_end"),
    "body": ("body_begin", "publish_end"),
    "empty": ("empty_begin", "empty_end"),
}
HELPERS = r'''
struct Bf16ProbeRecord {
  unsigned long long begin, end;
  unsigned task, wave, xcd, row;
};
struct Bf16Probe {
  unsigned long long begin = 0, end = 0;
  bool selected, started = false, ended = false;
  // compiler barrier bounds issue placement, without retiring memory operations
  static __device__ __forceinline__ unsigned long long tick() {
    unsigned long long t;
    asm volatile("s_memrealtime %0" : "=s"(t) :: "memory");
    return t;
  }
  __device__ __forceinline__ void first() { if (selected && !started) { begin = tick(); started = true; } }
  __device__ __forceinline__ void last() { if (selected && started && !ended) { end = tick(); ended = true; } }
  __device__ __forceinline__ void save(Bf16ProbeRecord* out, unsigned task) {
    if (selected && threadIdx.x % 64 == 0) {
      *out = {begin, end, task, (unsigned)etx_wave(), ETX_DOMAIN_ID(), BF16_PROBE_ROW};
    }
  }
};
'''


def replace_one(src, old, new):
    require(src.count(old) == 1, "source_anchor")
    return src.replace(old, new)


def current_source():
    """The performance control is byte-for-byte the unmodified emitter output."""
    return gemv.emit_tiles()


def specialised_source():
    """Up only: fixed K=2048, 32 gate and 32 up rows, 16 whole-row waves.

    Each row preserves the wave sum and all virtual zero additions, including
    the original partial's +0. The gate/up join is retained before SiLU. Down
    uses the original emitter and receives no up ownership proof.
    """
    s = current_source()
    start = s.index('template <int EPL, int U, int NB, class Epi>')
    end = s.index('template <int EPL, int U, class Epi>', start)
    f = s[start:end].replace('dsl_gemv_rows_nb', 'bf16_single_rows_nb')
    f = replace_one(f, '  const int per = (S + nw - 1) / nw, sb = wv * per, se = min(S, sb + per);',
        '  const int per = 16, sb = wv * per, se = sb + per;')
    f = replace_one(f, '  for (int i = (int)threadIdx.x; i < nw * nrt * NB; i += blockDim.x) pl[i] = 0.f;\n  __syncthreads();', '')
    # WSUM and fallback are preprocessor alternatives: preserve each original wave sum.
    f = f.replace('if (lane == 0) pl[(wv * nrt + cur) * NB + b] += v;',
        'if (lane == 0) { float z = 0.f; for (int w2 = 0; w2 < 16; ++w2) z += w2 == wv ? (0.f + v) : 0.f; epi(r0 + cur, &z); }')
    a = f.index('#ifdef DSL_ABLATE_NORED\n#pragma unroll\n  for (int b')
    b = f.index('#undef DSL_WP', a)
    f = f[:a] + '  __syncthreads(); // all gate/up results visible before SiLU; no LDS reuse before this join\n' + f[b:]
    f = f.replace('  typedef typename dsl_wvec<EPL>::T V;',
        '  static_assert(EPL == 8 && NB == 1, "fixed up mapping only");\n  typedef typename dsl_wvec<EPL>::T V;')
    s = s[:end] + f + s[end:]
    old = 'if (d.K % (ETX_WAVE_SIZE * 8) == 0 && 2 * gn <= DSL_RCH && gp1 % gspr == 0 && gp2 % gspr == 0 && (gn * gspr) % gp2 == 0) {'
    require(s.count(old) == 2, "specialisation_anchors")  # up and up_fold
    new = '''if (d.K == 2048 && gn == 32 && d.rpt == 32 && gnw == 16 && blockDim.x == 1024) {
    bf16_single_rows_nb<8, DSL_UW, 1>(We, d.K, 0, xs, r0, r0 + 2 * gn, part, [&](int r, const float* v) {
      const int q = r - r0; g[q < gn ? q : d.rpt + q - gn] = dsl_bfr(v[0]); }, gn, (long long)(d.M - gn) * d.K);
  } else ''' + old
    return s.replace(old, new)


def probe_source(pair, task=0, wave=0, row=0):
    require(pair in PAIRS and task >= 0 and 0 <= wave < 16 and 0 <= row < 64, "probe_selection")
    # Instrument an independent clone, not the production module or performance TU.
    s = current_source()
    helpers_end = s.index('// ---- dsl_gemv:')
    helper = s[:helpers_end]
    names = r'dsl_(?:gemv_rows_nb|gemv_rows|gemv_nb|gemv_chunked_nb|gemv_chunked|gemv|stage_nb|stage)'
    # Add one explicit probe parameter through existing helpers. The body remains
    # the same arithmetic, load pipeline and barriers (no ablations).
    rx = re.compile(r'(' + names + r'(?:<[^\n;{}()]*>)?\()')
    helper = rx.sub(lambda m: m[1] + ('Bf16Probe& probe, ' if 'void ' in helper[max(0, m.start()-90):m.start()].split('\n')[-1] else 'probe, '), helper)
    # Forward declaration of stage_nb ends on the same line and is handled above.
    core_start = helper.index('template <int EPL, int U, int NB, class Epi>')
    core_end = helper.index('template <int EPL, int U, class Epi>', core_start)
    prefix, suffix = helper[:core_start], helper[core_end:]
    helper = helper[core_start:core_end]
    begin, end = PAIRS[pair]
    def mark(point):
        return ('probe.first();' if point == begin else '') + ('probe.last();' if point == end else '')
    def inject(old, point, before=False):
        nonlocal helper
        new = (mark(point) + '\n' + old) if before else (old + '\n' + mark(point))
        helper = replace_one(helper, old, new)
    inject('  for (int i = (int)threadIdx.x; i < nw * nrt * NB; i += blockDim.x) pl[i] = 0.f;', 'zero_begin', True)
    inject('  __syncthreads();\n#if defined(ETX_DSL_SCALAR) || defined(ETX_DSL_GU1)', 'zero_end')
    inject('  int s0 = sb;', 'pipeline_begin', True)
    inject('  if (sb < se) flush();', 'pipeline_end')
    helper = replace_one(helper, '  __syncthreads();\n  for (int r = (int)threadIdx.x; r < nrt; r += blockDim.x) {',
                         '  __syncthreads();\n' + mark('join') + '\n  for (int r = (int)threadIdx.x; r < nrt; r += blockDim.x) {')
    helper = replace_one(helper, '  __syncthreads();\n#endif\n#undef DSL_WP',
                         mark('gather') + '\n  __syncthreads();\n' + mark('reuse') + '\n#endif\n#undef DSL_WP')
    helper = replace_one(helper, '  auto flush = [&]() {', '  auto flush = [&]() {\n  if (cur == BF16_PROBE_ROW) { ' + mark('flush_begin') + ' }')
    helper = replace_one(helper, '#endif\n  };\n#ifdef ETX_DSL_SCALAR',
        '#endif\n  if (cur == BF16_PROBE_ROW) { ' + mark('flush_end') + ' }\n  };\n#ifdef ETX_DSL_SCALAR')
    helper = prefix + helper + suffix
    # The helper contains shared includes before etx_wave's definition.
    pos = helper.index(coh.HELPERS)
    helper = helper[:pos] + HELPERS + helper[pos:]
    out = [f'#define BF16_PROBE_ROW {row}', helper]
    for export, form in [('dsl_moe_up', 'silu_mul'), ('dsl_moe_down', 'scale')]:
        body = rx.sub(lambda m: m[1] + 'probe, ', gemv.BODIES[form])
        old = '  const __hip_bfloat16* We = d.W + (size_t)dsl_eid * d.estride;'
        # Up has a comment after this anchor, down does not.
        body = replace_one(body, old, old + '\n' + mark('routing'))
        # Staging completes after its original barrier; selected K is not BIGK.
        stage = re.search(r'  dsl_stage\(probe, [^\n]+;', body)[0]
        body = replace_one(body, stage, stage + '\n' + mark('stage'))
        out += [f'extern "C" __global__ __launch_bounds__(1024, 1) void bf16_300_probe_{export}(const unsigned char* a, int first, Bf16ProbeRecord* records) {{',
                '  etx_import_init(); __syncthreads();',
                f'  int blk = first + blockIdx.x; Bf16Probe probe; probe.selected = blk == {task} && etx_wave() == {wave};',
                mark('body_begin'),
                '#ifdef ETX_DSL_SCALAR',
                '  blk = etx_uniform(blk);',
                '  const int nblk = *(const int DSL_ARG_AS*)(uintptr_t)(a + DSL_GRID_OFF + 12);',
                '#else',
                '  const int nblk = *(const int*)(a + DSL_GRID_OFF + 12);',
                '#endif',
                '  if (blk >= nblk) return;',
                '  const DslArgs d = dsl_args(a); float* xs = (float*)etx_lds_arena;', mark('args'), body,
                mark('publish_begin'), '  ETX_RELEASE_DOMAIN(); __syncthreads();', mark('publish_end'),
                mark('empty_begin'), mark('empty_end'),
                '  probe.save(records + blockIdx.x, blk);', '}']
    return '\n'.join(out)
