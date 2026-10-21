"""Emit unchanged kept router/KEY32 with one isolated elected finalizer.

Compilation never invokes a GPU. Native checks use retained actual model tensors.
"""
from pathlib import Path
import os
import subprocess
import sys

from examples.experiment_program.protocol import require

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def source(probe=False):
    # A clean subprocess freezes the kept emitter switches independently of any
    # already-imported generic emitter in local tests or other record builders.
    env={k:v for k,v in os.environ.items() if not k.startswith(('ETX_DSL_','ETX_VL_','DSL_ABLATE_','ETX_ABL_'))}
    env.update(ETX_VL_ROUTER_NARROW='1',ETX_VL_NB='1',ETX_VL_DSL_EXT='0',PYTHONDONTWRITEBYTECODE='1')
    code='from etx.ops import gemv,topk; print(gemv.emit_tiles());print(topk.emit_tiles())'
    bodies=subprocess.check_output([sys.executable,'-c',code],env=env,cwd=ROOT,text=True)
    return (('#define G01_PROBE 1\n' if probe else '')+'#define ETX_IMPORT_LDS 32768\n#define ETX_IMPORT_SLOTS 16\n'
            '#include "etx_lowering.h"\n#include "etx/primitives.h"\n#include "etx/tile_util.h"\n'+bodies+(HERE/'kernel.hip').read_text())


def capture_manifest(directory):
    from examples.vllm_llm.producer_replay import read_capture
    m=read_capture(directory,actual=True)
    require(m['layers']==48 and len(m['positions'])==1 and m['positions'][0] in (0,1040),'G01 requires complete original position0/1040 captures')
    return m
