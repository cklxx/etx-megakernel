"""Explicit GLM build knobs. Defaults preserve the pre-sweep execution settings.

vgpr_budget is a planner estimate, NOT a compiler register limit or measurement.
waves_per_simd_max requests the LLVM amdgpu_waves_per_eu(1, max) bound;
compiler reports, not this request, determine resource feasibility. LDS is the
native FP32 input staging slab, not an unused scheduler reservation.
"""
import argparse
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Options:
    threads: int = 256
    vgpr_budget: int = 64
    waves_per_simd_max: int = 0
    cache_ab: int = 0
    routed_slc: int = 0
    shared_slc: int = 0
    row_tile: int = 1
    k_tile: int = 0
    lds_bytes: int = 0

    def __post_init__(self):
        for f in fields(self):
            if type(getattr(self, f.name)) is not int:
                raise ValueError(f'{f.name} must be an integer')
        if self.threads not in (64, 128, 256):
            raise ValueError('threads must be 64,128,256')
        if not 1 <= self.vgpr_budget <= 512:
            raise ValueError('vgpr_budget must be 1..512 (planner estimate only)')
        if not 0 <= self.waves_per_simd_max <= 8:
            raise ValueError('waves_per_simd_max must be 0..8 (0=compiler default)')
        if self.cache_ab not in (0, 1) or self.routed_slc not in (0, 1) or self.shared_slc not in (0, 1):
            raise ValueError('cache_ab and SLC settings must be 0 or 1')
        if not 1 <= self.row_tile <= 32:
            raise ValueError('row_tile must be 1..32')
        if not 0 <= self.k_tile <= 2147483647:
            raise ValueError('k_tile must be 0..2147483647')
        if not 0 <= self.lds_bytes <= 65536 or self.lds_bytes % 4:
            raise ValueError('lds_bytes must be a multiple of 4 in 0..65536')

    def to_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def header(self):
        names = ('cache_ab', 'routed_slc', 'shared_slc', 'row_tile', 'k_tile', 'lds_bytes')
        return '#pragma once\n' + ''.join(f'#define ETX_GLM_{n.upper()} {getattr(self, n)}\n' for n in names)

    def compiler_flags(self):
        # The generated entrypoint carries the waves attribute. All numerical
        # macros live in the generated header, never ephemeral shell flags.
        return ['-O3', '-std=c++17', '--offload-arch=gfx942', '-include', 'glm53_options.h']


def from_dict(value):
    if not isinstance(value, dict):
        raise ValueError('build options must be a JSON object')
    unknown = set(value) - {f.name for f in fields(Options)}
    if unknown:
        raise ValueError(f'unknown build options: {sorted(unknown)}')
    return Options(**value)


def load(path=None):
    return Options() if path is None else from_dict(json.loads(Path(path).read_text()))


def add_arguments(parser):
    parser.add_argument('--options', type=Path, help='explicit options JSON; absent means original defaults')
    for f in fields(Options):
        parser.add_argument('--' + f.name.replace('_', '-'), type=int, default=None)


def from_arguments(args):
    base = load(args.options).to_dict()
    base.update({f.name: getattr(args, f.name) for f in fields(Options) if getattr(args, f.name) is not None})
    return from_dict(base)


def inject(source, options):
    """GLM-local generated-source wrapper; never modifies global ETX codegen."""
    source = '#include "glm53_options.h"\n' + source
    if options.waves_per_simd_max:
        marker = 'extern "C" __global__ void __launch_bounds__('
        if marker not in source:
            raise ValueError('cannot apply compiler waves bound: entrypoint not found')
        source = source.replace(marker, f'extern "C" __global__ __attribute__((amdgpu_waves_per_eu(1, {options.waves_per_simd_max}))) void __launch_bounds__(')
    return source
