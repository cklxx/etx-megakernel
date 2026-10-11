"""CPU-only, lossless compiler evidence for verification gaps.

Accepts retained assembly/IR, never launches a device. Static layout order is
not executed order. Missing provenance is reported explicitly, never guessed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from . import isa_hops as isa


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernels(text):
    """Read all kernel functions, retaining instruction indices and raw lines."""
    names = re.findall(r'\.amdhsa_kernel\s+(\w+)', text)
    lines = text.splitlines()
    result = {}
    for name in names:
        first, last = isa.function_lines(lines, name)
        parsed = isa.parse('\n'.join(lines[first-1:last]), first)
        instructions = [i for i in parsed if i.mnem and i.mnem != 'implicit_def']
        resources = {}
        tail = text[text.index('.amdhsa_kernel ' + name):] if '.amdhsa_kernel ' + name in text else ''
        for key, value in re.findall(r'; (NumVgprs|TotalNumSgprs|ScratchSize|LDSByteSize|Occupancy): (\d+)', tail):
            if key in resources:
                break
            resources[key] = int(value)
        result[name] = dict(resources=resources, instructions=[dict(index=n, assembly_line=i.line,
            mnemonic=i.mnem, operands=i.ops, modifiers=i.mods, text=i.text.strip())
            for n, i in enumerate(instructions)])
    if not result:
        raise ValueError('no AMDHSA kernels in assembly')
    return result


def report(assembly, ir=None):
    assembly = Path(assembly)
    result = dict(schema=1, basis='static compiler evidence, not device execution or timing',
        assembly_sha256=digest(assembly), kernels=kernels(assembly.read_text()))
    if ir:
        result['ir_sha256'] = digest(ir)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('assembly', type=Path)
    p.add_argument('--ir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = report(a.assembly, a.ir)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print('VERIFY ' + json.dumps(dict(kernels=len(result['kernels']),
        instructions={k:len(v['instructions']) for k,v in result['kernels'].items()},
        output=str(a.output)), sort_keys=True))


if __name__ == '__main__':
    main()
