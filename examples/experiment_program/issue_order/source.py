"""Exact immutable-read moves in the original gfx942 math islands.

No new load, store, branch, arithmetic expression, reduction or join is emitted.
Each move is confined to a decode path dominated by the original length guard.
The source reference is never overwritten or replaced for certification.
"""
from collections import Counter
import hashlib
import re
from etx.ops import attnprep

ATTENTION_SHA = '759c3e026b360627ad8d727351466464b2d1d62cf22e6e8860bd5c0f0a068b41'
RECORDS = ('G06','G07','G08')
MARKERS = {
 'G06': ('const long long pos = dsl_ap_scalar(positions)', 'ulo = dsl_a_ld64'),
 'G07': ('tmps[idx] = tmp_out_ptr[lastj_offset];', 'max_logit = fmaxf(max_logit, __shfl_xor'),
 'G08': ('Vlocal[vtoken_depth][vhe_depth][vfetch_depth] = *v_fetch_ptr_16B', '_B16x8 tmp = *q_fetch_ptr_16B'),
}

def prep_source(): return attnprep.TILE

def statement_inventory(text):
    text = re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
    return Counter(re.findall(r'[A-Za-z_]\w*|\d+|[^\s]',text))

def move(text, begin, end, before):
    if text.count(begin) != 1 or text.count(end) != 1 or text.count(before) != 1:
        raise ValueError('source island anchor changed')
    start = text.index(begin); stop = text.index(end,start)
    chunk = text[start:stop]
    text = text[:start] + text[stop:]
    return text.replace(before,chunk+before,1)

def transform(record, text):
    original = text
    if record == 'G06':
        if text != prep_source(): raise ValueError('prep reference changed')
        # Keep vector/fallback and q/k/V guards. Uniform metadata is not read on
        # any additional wave or path. Only positions and k-wave slot move.
        pos = '      const long long pos = dsl_ap_scalar(positions);     // uniform, and outside the divergent branch\n'
        cosp = '      const __hip_bfloat16* cosp = cs + pos * HD;         // [cos (HD/2) | sin (HD/2)]\n'
        sinp = '      const __hip_bfloat16* sinp = cosp + HALF;\n'
        slot = '      const long long slot = isq ? 0 : dsl_ap_scalar(slot_mapping);\n'
        for chunk in (pos+cosp+sinp,slot):
            if text.count(chunk) != 1: raise ValueError('prep metadata island changed')
            text = text.replace(chunk,'',1)
        before='      unsigned long long ulo = 0, uhi = 0;\n'
        text=text.replace(before,pos+cosp+sinp+slot+before,1)
    elif record in ('G07','G08'):
        if hashlib.sha256(text.encode()).hexdigest() != ATTENTION_SHA:
            raise ValueError('attention reference hash changed')
        if record == 'G07':
            # Only the gfx942 reduce definition, not the gfx950 alternatives.
            start=text.index('// Grid: (num_heads, num_seqs).\ntemplate <typename scalar_t, typename OUTT, int HEAD_SIZE')
            stop=text.index('\n}',start)+2
            body=text[start:stop]
            body=move(body,'    const float* exp_sums_ptr = exp_sums +','  #pragma unroll\n    for (int i = 0; i < NPAR_LOOPS; i++) {\n      const auto partition_no = i * WARP_SIZE + threadIdx.x;\n      rescaled_exp_sum',
                      '    float max_logit = reg_max_logit[0];')
            body=move(body,'  const scalar_t* tmp_out_ptr =','  __syncthreads();\n','  if (warpid == 0) {\n')
            text=text[:start]+body+text[stop:]
        else:
            # V metadata/payload depends on the existing guarded sequence len,
            # block table and V cache, and not Q LDS or Q/K loads. Original
            # Q-staging barrier stays exactly where it was relative to Q users.
            start=text.index('  constexpr int VTOKENS_PER_LANE =')
            stop=text.index('  // calculate post qk mfma scale',start)
            chunk=text[start:stop]
            text=text[:start]+text[stop:]
            before='  // fetch Q in shared across warps and then write to registers\n'
            if before not in text: raise ValueError('attention Q anchor changed')
            text=text.replace(before,chunk+before,1)
    else: raise ValueError('unknown issue-order record')
    if text == original or statement_inventory(text) != statement_inventory(original):
        raise ValueError('source move is empty or changed an instruction/guard')
    return text
