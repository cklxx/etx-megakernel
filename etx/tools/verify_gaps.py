"""CPU-only, lossless compiler evidence for verification gaps.

Accepts retained assembly/IR, never launches a device. Static layout order is
not executed order. Missing provenance is reported explicitly, never guessed.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re

from . import isa_hops as isa


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def parse_isa(text, first_line=1):
    # Commas inside DPP/swizzle modifiers are not operand separators.
    protected=re.sub(r'(quad_perm:\[[^]]+\]|offset:swizzle\([^)]*\))',
        lambda m:m[0].replace(',', '__COMMA__'),text)
    parsed=isa.parse(protected,first_line)
    original=text.splitlines()
    for i in parsed:
        i.ops=[o.replace('__COMMA__',',') for o in i.ops]
        i.mods=i.mods.replace('__COMMA__',',')
        i.text=original[i.line-first_line]
    return parsed

def kernels(text):
    """Read all kernel functions, retaining instruction indices and raw lines."""
    names = re.findall(r'\.amdhsa_kernel\s+(\w+)', text)
    lines = text.splitlines()
    result = {}
    for name in names:
        first, last = isa.function_lines(lines, name)
        parsed = parse_isa('\n'.join(lines[first-1:last]), first)
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



def function_ir(text, name):
    m = re.search(r'(?ms)^define [^\n]*@' + re.escape(name) + r'\([^\n]*\n.*?^}', text)
    if not m:
        raise ValueError('missing kernel IR: ' + name)
    return m[0].splitlines()


def type_layout(ty, types):
    """Size/alignment for the explicitly supported gfx942 LDS types."""
    ty=ty.strip()
    if ty in types: return type_layout(types[ty],types)
    m=re.fullmatch(r'i(8|16|32|64)',ty)
    if m: return int(m[1])//8, int(m[1])//8
    m=re.fullmatch(r'\[(\d+) x (.*)\]',ty)
    if m:
        size,alignment=type_layout(m[2],types)
        return int(m[1])*size,alignment
    if ty.startswith('{') and ty.endswith('}'):
        offset,align=0,1
        for member in ty[1:-1].split(','):
            size,a=type_layout(member,types); offset=(offset+a-1)//a*a+size; align=max(align,a)
        return (offset+align-1)//align*align,align
    raise ValueError('unsupported LDS type: '+ty)


def lds_inventory(dump, name):
    """Recover the compiler's actual field mapping from its before/after pass dump.

    No presumed alphabetical layout. Match unchanged instruction skeletons,
    validate unique field ownership and exact type size. Unknown mappings fail.
    """
    before,after=dump.split('*** IR Dump After',1)
    types=dict(re.findall(r'(?m)^([%\w.]+) = type (.*)$',after))
    layout=types['%llvm.amdgcn.kernel.'+name+'.lds.t']
    fields=[f.strip() for f in layout[1:-1].split(',')]
    originals={m[0]:dict(type=m[1],alignment=int(m[2])) for m in
        re.findall(r'(?m)^@([^ ]+) = .*addrspace\(3\) global (.*?) (?:undef|poison), align (\d+)',before)}
    a=[line for line in function_ir(before,name) if any(n in originals for n in re.findall(r"@([^ ,)]+)",line))]
    b=[line for line in function_ir(after,name) if "@llvm.amdgcn.kernel."+name+".lds" in line]
    pat=r'getelementptr inbounds \(%llvm\.amdgcn\.kernel\.'+re.escape(name)+r'\.lds.t, ptr addrspace\(3\) @llvm\.amdgcn\.kernel\.'+re.escape(name)+r'\.lds, i32 0, i32 (\d+)\)'
    def skeleton(line):
        line=re.sub(pat,'@LDS',line)
        line=re.sub(r'@[^ ,)]+',lambda m:'@LDS' if m[0][1:] in originals or m[0].endswith('.lds') else m[0],line)
        line=re.sub(r', !.*','',line)
        line=re.sub(r'align \d+','align A',line)
        return re.sub(r'%\d+','%SSA',line)
    # Match only LDS references; retain their full source-relative order.
    mapping={}
    matcher=difflib.SequenceMatcher(None,list(map(skeleton,a)),list(map(skeleton,b)),autojunk=True)
    for match in matcher.get_matching_blocks():
        for x,y in zip(a[match.a:match.a+match.size],b[match.b:match.b+match.size]):
            names=[n for n in re.findall(r'@([^ ,)]+)',x) if n in originals]
            indexes=[int(n) for n in re.findall(pat,y)]
            if names and not indexes and '@llvm.amdgcn.kernel.'+name+'.lds' in y: indexes=[0]
            if len(names)==len(indexes)==1:
                n,index=names[0],indexes[0]
                if index in mapping and mapping[index]!=n: raise ValueError('ambiguous LDS mapping')
                mapping[index]=n
    # A repeated skeleton can be removed by SequenceMatcher's popularity
    # heuristic. Recover it only when both sides name ONE unique owner/field.
    original_groups={}; lowered_groups={}
    for line in a:
        owners={n for n in re.findall(r'@([^ ,)]+)',line) if n in originals}
        original_groups.setdefault(skeleton(line),set()).update(owners)
    for line in b:
        indexes={int(n) for n in re.findall(pat,line)}
        if not indexes and '@llvm.amdgcn.kernel.'+name+'.lds' in line: indexes={0}
        lowered_groups.setdefault(skeleton(line),set()).update(indexes)
    for key,owners in original_groups.items():
        indexes=lowered_groups.get(key,set())
        if len(owners)==len(indexes)==1:
            index=next(iter(indexes)); owner=next(iter(owners))
            if index in mapping and mapping[index]!=owner: raise ValueError('conflicting LDS ownership')
            mapping[index]=owner
    # A uniquely sized remaining object/field is also an exact bijection,
    # provided every other field already has a verified owner.
    remaining={n for line in a for n in re.findall(r'@([^ ,)]+)',line) if n in originals} - set(mapping.values())
    for index,ty in enumerate(fields):
        if index in mapping: continue
        size,_=type_layout(ty,types)
        owners=[n for n in remaining if type_layout(originals[n]['type'],types)[0]==size]
        possible_fields=[j for j,f in enumerate(fields) if j not in mapping and type_layout(f,types)[0]==size]
        if len(owners)==len(possible_fields)==1:
            mapping[index]=owners[0]; remaining.remove(owners[0])
    if len(mapping)!=len(fields): raise ValueError(f'incomplete LDS mapping {name}: {mapping}')
    def construct(symbol):
        if symbol=='etx_lds_arena': return 'etx/import.h: __shared__ aligned(16) unsigned char etx_lds_arena[ETX_IMPORT_LDS]'
        if symbol in ('etx_imp_bar_count','etx_imp_bar_gen'): return 'etx/import.h: __shared__ uint32_t '+symbol+'[ETX_IMPORT_SLOTS]'
        if symbol=='etx_imp_nbar': return 'etx/import.h: __shared__ uint32_t etx_imp_nbar[16]'
        if symbol in ('etx_imp_done','etx_imp_exit'): return 'etx/import.h: __shared__ uint32_t '+symbol
        if re.match(r'_ZZ.*E7etx_lds$',symbol): return 'generated etx_run_task/etx_unfused: __shared__ aligned(16) unsigned char etx_lds[ETX_LDS_USED]'
        for label in ('s_task','s_tid','s_domain','s_worker'):
            if symbol.endswith(label): return 'generated etx_megakernel: __shared__ '+('etx_task ' if label=='s_task' else 'int32_t ')+label
        return 'compiler/global object; source attribution unknown'
    objects=[]; offset=0
    for i,ty in enumerate(fields):
        size,align=type_layout(ty,types)
        obj=mapping[i]; declared=originals[obj]
        dsize,_=type_layout(declared['type'],types)
        if size!=dsize: raise ValueError('LDS field size mismatch')
        align=max(align,declared['alignment']); start=(offset+align-1)//align*align
        objects.append(dict(symbol=obj,field=i,bytes=size,range=[start,start+size],
            padding_before=start-offset,construct=construct(obj),
            mapping_basis='before/after amdgpu-lower-module-lds instruction matching; unique-size bijection for unmatched fields'))
        offset=start+size
    return dict(objects=objects,total_bytes=offset,
        compiler_promoted_objects=[o['symbol'] for o in objects if o['construct']=='compiler/global object; source attribution unknown'],
        range_convention='half-open byte ranges; alignment padding explicit')


def dependencies(ins):
    """Explicit and implicit register uses/defs for the supported gfx9 syntax."""
    m,ops=ins.mnem,ins.ops
    allregs=lambda seq: {r for op in seq for r in isa.regs(op)}
    if m.startswith(('s_cmp','s_bitcmp')): return {'scc'},allregs(ops)
    if m.startswith('s_cbranch'):
        c=m[len('s_cbranch_'):]; return set(),{'exec' if c.startswith('exec') else 'vcc' if c.startswith('vcc') else 'scc'}
    if isa._NODEF.match(m): return set(),allregs(ops)
    defs=allregs(ops[:1]); uses=allregs(ops[1:])
    if m.startswith('v_cmp') and len(ops)==2:
        defs={'vcc'}; uses=allregs(ops)
    if m.startswith('v_cmpx'): defs.add('exec')
    if isa._CARRY_OUT.search(m) and len(ops)>1 and any(r.startswith('s') or r=='vcc' for r in isa.regs(ops[1])):
        defs.update(isa.regs(ops[1])); uses=allregs(ops[2:])
    if m.startswith('s_') and isa._SALU_SCC.match(m): defs.add('scc')
    if m.startswith('v_') or ins.kind.startswith(('vm_','lds_')): uses.add('exec')
    if 'saveexec' in m: defs.update(('exec','scc')); uses.add('exec')
    if m.startswith(('v_cndmask','v_addc','v_subb')): uses.add('vcc')
    return defs,uses


def memory_order(instructions):
    """Unfiltered memory/wait/barrier sequence and same-block independence sites.

    Independence is register data/control independence, not an alias/faulting
    proof or permission to remove a wait. Branches, barriers and stores terminate
    candidate windows. Counts are sites, not executed loads or cycles.
    """
    order=[]; candidates=[]; pending={'vmcnt':[], 'lgkmcnt':[]}; drained={}
    ancestry={}
    for row in instructions:
        ins=isa.Insn(row['assembly_line'],row['mnemonic'],row['operands'],row['modifiers'],row['text'])
        m=ins.mnem; defs,uses=dependencies(ins); origin=set().union(*(ancestry.get(r,set()) for r in uses))
        mem=ins.kind!='alu' or m.startswith(('v_readlane','v_writelane'))
        if mem or m.startswith(('s_waitcnt','s_barrier')):
            order.append(dict(**row,wait_scope={k:int(v) for k,v in re.findall(r'(vmcnt|lgkmcnt|expcnt)\((\d+)\)',row['text'])}))
        if m.startswith('s_cbranch') or m in ('s_branch','s_barrier','s_endpgm') or ins.kind.endswith('store'):
            pending={'vmcnt':[], 'lgkmcnt':[]}; drained={}; ancestry={}
        if m=='s_waitcnt':
            for counter,count in re.findall(r'(vmcnt|lgkmcnt)\((\d+)\)',row['text']):
                n=int(count); queue=pending[counter]
                completed=queue[:-n] if n else queue[:]
                drained[counter]=(row['index'],completed)
                pending[counter]=queue[-n:] if n else []
        counter='vmcnt' if ins.kind=='vm_load' else 'lgkmcnt' if ins.kind in ('scalar_load','lds_load') else None
        if counter:
            prior=drained.get(counter)
            if prior:
                wait,loads=prior
                independent=[l for l in loads if l not in origin]
                if independent:
                    candidates.append(dict(load=row['index'],wait=wait,independent_prior_loads=independent,
                        status='register-independent candidate; alias/range/reorder legality not established'))
            pending[counter].append(row['index']); origin.add(row['index'])
        for r in defs: ancestry[r]=origin.copy()
    return dict(sequence=order,register_independent_load_sites=len(candidates),candidates=candidates,
        proven_overlapable_loads=None,
        limitation='layout order and same-block register independence do not establish legal hoisting or exposed time')


def annotate_sources(assembly, kernel):
    files={}; source=None; mapping={}
    for n,line in enumerate(assembly.splitlines(),1):
        m=re.match(r'\s*\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?',line)
        if m: files[m[1]]=str(Path(m[2])/m[3]) if m[3] else m[2]
        m=re.match(r'\s*\.loc\s+(\d+)\s+(\d+)',line)
        if m: source=dict(file=files.get(m[1]),line=int(m[2]))
        mapping[n]=source
    for row in kernel['instructions']: row['source']=mapping.get(row['assembly_line'])

def report(assembly, ir=None, lds_dump=None):
    assembly = Path(assembly)
    result = dict(schema=1, basis='static compiler evidence, not device execution or timing',
        assembly_sha256=digest(assembly), kernels=kernels(assembly.read_text()))
    for kernel in result['kernels'].values(): annotate_sources(assembly.read_text(),kernel)
    if ir:
        result['ir_sha256'] = digest(ir)
    for name,kernel in result['kernels'].items():
        kernel['memory_order']=memory_order(kernel['instructions'])
        if lds_dump:
            kernel['lds']=lds_inventory(Path(lds_dump).read_text(),name)
            if kernel['lds']['total_bytes']!=kernel['resources']['LDSByteSize']:
                raise ValueError('LDS inventory does not reconcile with assembly')
    if lds_dump: result['lds_dump_sha256']=digest(lds_dump)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('assembly', type=Path)
    p.add_argument('--ir', type=Path)
    p.add_argument('--lds-dump', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = report(a.assembly, a.ir, a.lds_dump)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print('VERIFY ' + json.dumps(dict(kernels=len(result['kernels']),
        instructions={k:len(v['instructions']) for k,v in result['kernels'].items()},
        register_independent_load_sites={k:v['memory_order']['register_independent_load_sites'] for k,v in result['kernels'].items()},
        output=str(a.output)), sort_keys=True))


if __name__ == '__main__':
    main()
