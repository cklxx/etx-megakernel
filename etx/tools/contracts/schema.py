"""Schema: acquire contract."""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from etx.tools.contracts.source_control import compact


ROOT = Path(__file__).resolve().parents[3]


class Refusal(Exception):
    pass


@dataclass(frozen=True)
class Param:
    offset: int
    ty: str            # "ptr" (addrspace 1) or "i32"
    name: str
    binding: str       # the host expression (etx/importer/spec.py ARGUMENTS)
    access: str = ""   # ptr only: "read", "write" or "none"
    noalias: bool = False
    kinds: frozenset = frozenset()   # load / store kinds the body must show: plain, nontemporal, memcpy, store
    role: str = ""     # weight, activation, step-input, null


def P(offset, name, binding, access, noalias, kinds, role):
    return Param(offset, "ptr", name, binding, access, noalias, frozenset(kinds), role)


def I(offset, name, binding):
    return Param(offset, "i32", name, binding)


PRODUCER, CONSUMER = "attn_reduce", "wvsplitk_y2"


PRODUCER_TILE, CONSUMER_TILE = "etx_tile_attn_reduce", "etx_tile_wvsplitk_y2"


CROSS = "attn_out"


CONTRACT = {
    PRODUCER: (
        P(0, "out", "attn_out", "write", True, {"store"}, "activation"),
        P(8, "exp_sums", "exp_sums", "read", True, {"plain"}, "activation"),
        P(16, "max_logits", "max_logits", "read", True, {"plain"}, "activation"),
        P(24, "tmp_out", "tmp_out", "read", True, {"plain"}, "activation"),
        P(32, "seq_lens", "seq_lens", "read", True, {"plain"}, "step-input"),
        P(40, "ptr5", "nullptr", "read", True, {"plain"}, "null"),
        I(48, "max_num_partitions", "parts"),
        P(56, "ptr7", "nullptr", "read", True, {"plain"}, "null"),
    ),
    CONSUMER: (
        I(0, "K", "x.K"), I(4, "Kbp", "x.K"), I(8, "Kap", "x.K"), I(12, "M", "xM"), I(16, "Bx", "b ? xM : 1"),
        I(20, "By", "1"),
        P(24, "B", "w", "read", False, {"nontemporal"}, "weight"),
        P(32, "A", "in", "read", True, {"memcpy"}, "activation"),
        P(40, "BIAS", "b", "read", True, {"plain"}, "null"),
        P(48, "C", "out", "write", False, {"store"}, "activation"),
        I(56, "WvPrGrp", "x.wv"), I(60, "CuCount", "x.gx * x.gy * x.gz"),
    ),
}


GRID_OFFSETS = (64, 68, 72)


ETX_PARAMS = ("etx.bx", "etx.by", "etx.bz", "etx.gx", "etx.gy", "etx.gz", "etx.tid", "etx.slot")


CROSS_USERS = {("attn", 16): "none", ("reduce", 0): "write"}


ATTN_EXPORT_PREFIX = "attn_mfma"


OPCODES = {
    "add", "sub", "mul", "udiv", "sdiv", "urem", "srem", "shl", "lshr", "ashr", "and", "or", "xor",
    "fadd", "fsub", "fmul", "fdiv", "fneg", "zext", "sext", "trunc", "fpext", "fptrunc", "sitofp", "uitofp",
    "fptosi", "fptoui", "bitcast", "icmp", "fcmp", "select", "phi", "br", "ret", "freeze", "getelementptr",
    "extractelement", "insertelement", "shufflevector", "load", "store", "call", "fence", "unreachable",
}


CALLEES = {
    "llvm.umin.i32", "llvm.umax.i32", "llvm.smin.i32", "llvm.smax.i32", "llvm.maxnum.f32", "llvm.minnum.f32",
    "llvm.exp.f32", "llvm.exp2.f32", "llvm.amdgcn.exp2.f32", "llvm.fabs.f32", "llvm.amdgcn.s.barrier",
    "llvm.amdgcn.wave.barrier", "llvm.amdgcn.sched.barrier", "llvm.amdgcn.update.dpp.i32",
    "llvm.amdgcn.mfma.f32.4x4x4bf16.1k", "llvm.amdgcn.mbcnt.lo", "llvm.amdgcn.mbcnt.hi",
    "llvm.amdgcn.ds.bpermute", "llvm.memcpy.p3.p1.i64", "llvm.lifetime.start.p5", "llvm.lifetime.end.p5",
    "llvm.assume", "etx_imp_slot_barrier",
}


WG_FENCE = re.compile(r'^fence syncscope\("workgroup"\) (acquire|release)$')


ALLOWED_DEFINES = {"ETX_DSL_NT", "ETX_DSL_VECNORM", "ETX_VL_CHECK", "ETX_DSL_PREP_EARLY", "ETX_DSL_UNIFORM",
                   "ETX_NO_PRE_WAIT", "ETX_NO_TOUCH", "ETX_BACKOFF_STMT"}


LOWERING = {
    "ETX_RELEASE_DOMAIN()": 'do { asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); } while (0)',
    "ETX_RELEASE_DEVICE()": 'do { asm volatile("buffer_wbl2 sc1\\n\\ts_waitcnt vmcnt(0)" ::: "memory"); } while (0)',
    "ETX_ACQUIRE_DEVICE()": 'do { asm volatile("buffer_inv sc1" ::: "memory"); } while (0)',
    "ETX_POLL_DEVICE(ptr)": "(__hip_atomic_load((ptr), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))",
    "ETX_ARRIVE_DEVICE(ptr)": "(__hip_atomic_fetch_sub((ptr), 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))",
}


ARRIVE_FLUSH = r"""
static __device__ __forceinline__ int32_t etx_arrive_flush_DEVICE_s(const etx_params& p, int32_t idx, uint32_t domain, int32_t share) {
  if (!p.ev_share || share <= 1) { ETX_RELEASE_DEVICE(); return ETX_ARRIVE_DEVICE(p.events + ETX_EVW(idx)) - 1; }
  ETX_RELEASE_DOMAIN();
  const int32_t old = atomicAdd(p.ev_sub + (size_t)domain * p.event_words + idx, 1);
  if (old + 1 != share) return 1;
  ETX_RELEASE_DEVICE();
  return etx_atomic_sub_device(p.events + ETX_EVW(idx), share) - share;
}
"""


ATOMIC_SUB_DEVICE = r"""
static __device__ __forceinline__ int32_t etx_atomic_sub_device(int32_t* p, int32_t n) {
  return __hip_atomic_fetch_sub(p, n, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
"""


WAIT_SCOPE = r"""
  static __device__ __forceinline__ void etx_wait_##S(etx_event* e, const int32_t* abort_flag) {
    uint32_t n = 0;
    while (ETX_POLL_##S(e) > 0) {
      ETX_BACKOFF();
      if ((++n & 255u) == 0 && ETX_POLL_DEVICE(abort_flag) != 0) return;
    }
  }
"""


PRE_WAIT = ('extern "C" __device__ __attribute__((always_inline)) void etx_pre_wvsplitk_y2(const etx_ctx* c) '
            '{ etx_imp_touch(etx_imp_argblock(c->args, c->cst[0]), 128); }')


IMP_TOUCH = r"""
static __device__ __forceinline__ void etx_imp_touch(const unsigned char* a, int bytes) {
  const __attribute__((address_space(4))) uint32_t* a4 = (const __attribute__((address_space(4))) uint32_t*)(uintptr_t)a;
  uint32_t acc = a4[(bytes - 4) / 4];
  for (int o = 0; o < bytes; o += 64) acc ^= a4[o / 4];
  asm volatile("" :: "s"(acc));
}
"""


@dataclass
class Function:
    name: str
    header: str
    params: list            # (type, attrs tuple, name)
    lines: list = field(default_factory=list)
