"""CFG-aware conservative register/slot dependence, without device timing.

A chain is the MAY-dependence slice from non-scratch scalar/global loads to
any final DSL wave-sum lane-pair add (quad_perm:[1,0,3,2]). Reaching definitions
merge all CFG predecessors, including backedges. Explicit registers, implicit
exec/vcc/scc, compiler SGPR lanes and scratch words are tracked. LDS aliasing
is conservative: all LDS writes may feed every LDS read. This is not an
executed longest path, source variable recovery, or a timing critical path.
"""
from collections import Counter
import re

from etx.tools.isa import hops as isa
from etx.tools.diagnostics.verify_gaps import dependencies, parse_isa


def analyze(assembly, name, reserved):
    lines=assembly.splitlines(); first,last=isa.function_lines(lines,name)
    parsed=parse_isa('\n'.join(lines[first-1:last]),first)
    parsed=[i for i in parsed if not (not i.mnem and i.text.strip().startswith('.Ltmp'))]
    cfg=isa.blocks(parsed,{i+1:l for i,l in enumerate(lines)})
    rows=[i for i in parsed if i.mnem and i.mnem!='implicit_def']
    index={i.line:n for n,i in enumerate(rows)}
    graph={}; ends={}; phis={}; next_phi=-1; stores={}; reloads={}; roots=set(); sinks=set(); final_moves=set(); add_sites=set()
    predecessors={b:[] for b in range(len(cfg))}
    for b,block in enumerate(cfg):
        for succ in block.succ: predecessors[succ].append(b)
    def phi(block,key):
        nonlocal next_phi
        if (block,key) not in phis: phis[block,key]=next_phi; next_phi-=1
        return phis[block,key]
    def slot(ins):
        if ins.mnem.startswith(('v_readlane','v_writelane')):
            carrier=ins.ops[0] if ins.mnem.startswith('v_writelane') else ins.ops[1]
            if carrier not in reserved: return None
            return ['lane:'+carrier+':'+ins.ops[2]]
        if ins.kind not in ('scratch_store','scratch_load'): return None
        offset=re.search(r'offset:(\d+)',ins.text); base=int(offset[1]) if offset else 0
        width=re.search(r'dwordx(\d+)',ins.mnem); width=int(width[1]) if width else 1
        return ['scratch:'+str(base+i*4) for i in range(width)]
    for b,block in enumerate(cfg):
        state={}
        def read(key): return state[key] if key in state else phi(b,key)
        for ins in block.insns:
            if ins.line not in index: continue
            n=index[ins.line]; defs,uses=dependencies(ins); keys=slot(ins)
            is_store=keys and (ins.kind=='scratch_store' or ins.mnem.startswith('v_writelane'))
            is_reload=keys and not is_store
            if keys:
                if is_store:
                    data=ins.ops[1] if ins.kind=='scratch_store' else ins.ops[1]
                    uses=set(isa.regs(data)); defs=set(keys)
                    stores[n]=dict(index=n,assembly_line=ins.line,slots=keys,value_registers=sorted(uses),instruction=ins.text.strip())
                else:
                    defs=set(isa.regs(ins.ops[0])); uses=set(keys)
                    reloads[n]=dict(index=n,assembly_line=ins.line,slots=keys,value_registers=sorted(defs),instruction=ins.text.strip())
            elif ins.mnem.startswith('v_writelane'):
                # Ordinary lane insertion reads the other unchanged lanes.
                uses.update(defs)
            if ins.kind=='lds_store':
                uses.add('LDS_may_alias'); defs.add('LDS_may_alias')
            elif ins.kind=='lds_load': uses.add('LDS_may_alias')
            graph[n]={read(k) for k in uses}
            for k in defs: state[k]=n
            if ins.kind in ('vm_load','scalar_load') and not keys: roots.add(n)
            if ins.mnem.startswith('v_add_f32'): add_sites.add(n)
            if 'quad_perm:[1,0,3,2]' in ins.text.replace(' ',''):
                if ins.mnem.startswith('v_add_f32'): sinks.add(n)
                else: final_moves.add(n)
        ends[b]=state
    control,unreachable=control_dependencies(cfg)
    for b,block in enumerate(cfg):
        branch_nodes={index[line] for line in control[b] if line in index}
        for ins in block.insns:
            if ins.line in index: graph[index[ins.line]].update(branch_nodes)
    # Phi nodes implement reaching definitions without flattening loop order.
    pending=list(phis)
    visited=set()
    while pending:
        b,key=pending.pop()
        if (b,key) in visited: continue
        visited.add((b,key)); edges=set()
        for pred in predecessors[b]:
            if key in ends[pred]: edges.add(ends[pred][key])
            else:
                edges.add(phi(pred,key))
                if (pred,key) not in visited: pending.append((pred,key))
        graph[phis[b,key]]=edges
    # LLVM can defer the final addition into the lane-zero branch. Follow
    # reaching-register phi nodes to the final pair shuffle across that edge.
    for n in add_sites:
        todo=list(graph[n]); seen=set()
        while todo:
            x=todo.pop()
            if x in seen: continue
            seen.add(x)
            if x in final_moves: sinks.add(n)
            elif x<0: todo.extend(graph.get(x,()))
    def closure(start,edges):
        done=set(); todo=list(start)
        while todo:
            n=todo.pop()
            if n in done:continue
            done.add(n); todo.extend(edges.get(n,()))
        return done
    back=closure(sinks,graph)
    reverse={}
    for n,edges in graph.items():
        for src in edges: reverse.setdefault(src,set()).add(n)
    forward=closure(roots,reverse)
    chain=back & forward
    # Pair a reload with all CFG-reaching stores of each concrete spill slot.
    # Traverse only slot phi nodes here, not the spilled value's ancestors.
    for n,row in reloads.items():
        reaching=set(); todo=list(graph[n]); seen=set()
        while todo:
            x=todo.pop()
            if x in seen:continue
            seen.add(x)
            if x in stores: reaching.add(x)
            elif x<0: todo.extend(graph.get(x,()))
        row['reaching_stores']=sorted(reaching)
        row['value']=' / '.join(f"{r}@reload{n}" for r in row['value_registers'])
        row['on_may_chain']=n in chain
        row['executed_critical_chain']=None
        row['pair_complete']=set(row['slots']) <= {slot for x in reaching for slot in stores[x]['slots']}
    for n,row in stores.items():
        todo=list(graph[n]); seen=set(); definitions=set()
        while todo:
            x=todo.pop()
            if x in seen:continue
            seen.add(x)
            if x>=0:definitions.add(x)
            else:todo.extend(graph.get(x,()))
        row['reaching_value_and_control_definitions']=sorted(definitions)
    counts=Counter('scratch' if r['slots'][0].startswith('scratch') else 'sgpr_lane' for n,r in reloads.items() if n in chain)
    return dict(definition=__doc__.strip(),stores=list(stores.values()),reloads=list(reloads.values()),
        may_chain_reload_counts=dict(scratch=counts['scratch'],sgpr_lane=counts['sgpr_lane'],total=sum(counts.values())),
        reload_counts=dict(scratch=sum(r['slots'][0].startswith('scratch') for r in reloads.values()),
                           sgpr_lane=sum(r['slots'][0].startswith('lane') for r in reloads.values())),
        unmatched_reload_indices=[n for n,r in reloads.items() if not r['pair_complete']],
        final_reduction_indices=sorted(sinks),load_roots=len(roots),
        executed_critical_reload_count=None,
        blocks_without_exit_path=unreachable,
        limitation='may-dependence count is an upper-set static classification, not exposed latency or a single executed critical path')


def reserved_from_mir(text,name):
    match=re.search(r'(?ms)^name:\s+'+re.escape(name)+r'\s*\n(.*?)(?:^\.\.\.|\Z)',text)
    if not match: raise ValueError('missing allocated MIR kernel '+name)
    r=re.search(r'wwmReservedRegs:\s*\n((?:\s+-[^\n]*\n)*)',match[1])
    return ['v'+n for n in re.findall(r'\$vgpr(\d+)',r[1] if r else '')]


def control_dependencies(cfg):
    """Compute branch control dependence using immediate postdominators.

    Cooper's iterative dominator algorithm runs on the reversed CFG with one
    virtual exit; a branch's condition ends at its immediate postdominator.
    """
    exit_node=len(cfg); reverse={k:[] for k in range(exit_node+1)}
    successors={k:list(b.succ) for k,b in enumerate(cfg)}
    for k,b in enumerate(cfg):
        if b.exit or not b.succ: successors[k].append(exit_node)
        for v in successors[k]: reverse[v].append(k)
    seen=set(); post=[]; stack=[(exit_node,False)]
    while stack:
        n,done=stack.pop()
        if done: post.append(n); continue
        if n in seen:continue
        seen.add(n); stack.append((n,True))
        stack.extend((v,False) for v in reverse[n] if v not in seen)
    order=list(reversed(post)); rank={n:i for i,n in enumerate(order)}
    parent={exit_node:exit_node}
    def intersect(a,b):
        while a!=b:
            while rank[a]>rank[b]:a=parent[a]
            while rank[b]>rank[a]:b=parent[b]
        return a
    changed=True
    while changed:
        changed=False
        for n in order[1:]:
            pred=[p for p in successors[n] if p in parent]
            if not pred:continue
            new=pred[0]
            for p in pred[1:]:new=intersect(new,p)
            if parent.get(n)!=new:parent[n]=new;changed=True
    result={k:set() for k in range(exit_node)}
    for b,block in enumerate(cfg):
        last=block.insns[-1] if block.insns else None
        if not last or not last.mnem.startswith('s_cbranch') or b not in parent:continue
        for succ in block.succ:
            runner=succ; visited=set()
            while runner!=parent[b] and runner in parent and runner not in visited:
                visited.add(runner); result[runner].add(last.line);runner=parent[runner]
    return result, sorted(set(range(exit_node))-seen)
