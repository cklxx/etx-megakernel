"""gfx942 persistent capacity from the actual native unified VGPR descriptor.

LLVM22 AMDHSA vgpr_count already includes the accumulator allocation. The
separate agpr_count is descriptive and must not be added a second time.
"""
from __future__ import annotations
import json,re
from .protocol import require,resolve_path,sha


def register_allocation(metadata,next_free,rsrc1):
    # The linked descriptor is authoritative. LLVM may reserve additional call
    # ABI registers beyond metadata's named VGPR/AGPR population (G13:96/97).
    require(0<=metadata<=next_free,'metadata unified VGPR count exceeds actual descriptor')
    allocated=((next_free+7)//8)*8
    require(((rsrc1&63)+1)*8==allocated,'linked RSRC1 allocation differs from assembly unified VGPR count')
    return allocated,next_free-metadata


def persistent_capacity(directory,kernel='etx_megakernel_d0'):
    from examples.vllm_llm.bf16_300.build import resources
    from etx.machine.model import load_machine
    from .row_ops.elf import descriptors
    d=resolve_path(directory);assembly=(d/'mk.s').read_text();plan=json.loads((d/'plan.json').read_text())
    require('gfx942' in assembly,'only pinned gfx942 descriptor capacity implemented')
    matches=[r for r in resources(assembly) if r['kernel']==kernel];require(len(matches)==1,'exact persistent kernel metadata required')
    raw={k:v for k,v in matches[0].items() if k not in ('numeric_gate','static_blocks_per_cu','actual_co_residency')}
    match=re.search(r'(?ms)^\s*\.amdhsa_kernel\s+'+re.escape(kernel)+r'\s*$(.*?)^\s*\.end_amdhsa_kernel\s*$(.*?); Occupancy:\s*(\d+)',assembly)
    require(match is not None,'actual persistent kernel descriptor/occupancy missing')
    next_vgpr=re.search(r'\.amdhsa_next_free_vgpr\s+(\d+)',match[1]);require(next_vgpr is not None,'missing actual next_free_vgpr')
    next_vgpr=int(next_vgpr[1])
    physical=descriptors(d/'model.hsaco');require(kernel in physical,'linked persistent kernel descriptor missing')
    descriptor=physical[kernel]
    allocated,extra_registers=register_allocation(raw['vgpr'],next_vgpr,descriptor['rsrc1'])
    require(descriptor['group_segment']==raw['lds_bytes'] and descriptor['private_segment']==raw['scratch_bytes_lane'],'linked descriptor LDS/scratch differs from metadata')
    machine=load_machine('gfx942');caps=machine.resources;threads=plan['threads'];waves=(threads+63)//64
    compiler_waves=int(match[3]);register_waves=min(caps['max_waves_per_simd'],caps['regs_per_lane']//max(1,allocated))
    require(0<compiler_waves<=register_waves,'compiler occupancy inconsistent with physical unified registers')
    resident=min(caps['max_wg_per_cu'],65536//max(1,((raw['lds_bytes']+511)//512)*512),caps['simds_per_cu']*compiler_waves//waves)
    required=(plan['workers_per_domain']+machine.cus_per_domain()-1)//machine.cus_per_domain()
    return dict(raw_metadata=raw,next_free_vgpr=next_vgpr,descriptor_extra_registers=extra_registers,allocated_unified_registers_per_lane=allocated,register_allocation_quantum=8,agpr_already_in_unified_count=True,compiler_waves_per_simd=compiler_waves,threads=threads,waves_per_block=waves,workers_per_domain=plan['workers_per_domain'],resident_capacity=resident,required_persistent_blocks=required,linked_descriptor=descriptor,assembly_sha256=sha(d/'mk.s'),module_sha256=sha(d/'model.hsaco'),plan_sha256=sha(d/'plan.json'),gpu_residency='UNMEASURED')
