"""Resource-only plan: retain every kept tile and add the actual C6 body.

Do not run this plan with host.hip. It has an unbound extra C6 argument.
It measures the compiler envelope, not full-model C6 scheduling or correctness.
"""
import os
from pathlib import Path
from etx.frontends import hip_link
from etx.ir import Resource
from examples.vllm_llm import model


def build():
    g=model.build()
    out=Path(os.environ['C6_FOOTPRINT_OUT'])
    out.mkdir(parents=True,exist_ok=True)
    # The full model already includes the exact same prep and import adapters.
    src=out/'c6_body.hip'
    src.write_text((Path(__file__).parent/'probe.hip').read_text())
    g.tensor('c6_resource_args',(1,),dtype='i32',role='scratch')
    g.call_device('c6_resource_path',(304,),hip_link(str(src),'etx_tile_c6_resource'),
                  args=['c6_resource_args'],resource=Resource(threads=1024,vgpr=128,lds_bytes=8448),
                  in_edges={'E_argmax':'i->(0)'},domain_map='i->(i%8)',duration_us=1)
    return g


def bindings(**overrides): return {}


def runtime(bindings,rng=None): return {}
