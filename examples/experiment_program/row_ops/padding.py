"""Explicit G11 resource reserve matching; changes descriptors, never instructions."""
import hashlib,json,re,sys
from pathlib import Path
NAMES=('g11_control','g11_probe','g11_empty')
RX=re.compile(r'(?s)(\.amdhsa_kernel\s+(\S+)\n)(.*?)(\.end_amdhsa_kernel)')


def match(out):
    out=Path(out);raw={n:(out/(n+'.s')).read_text() for n in NAMES};limits={}
    for source in raw.values():
        for m in RX.finditer(source):
            count=int(re.search(r'\.amdhsa_next_free_sgpr (\d+)',m[3])[1]);limits[m[2]]=max(limits.get(m[2],0),count)
    if not limits:raise ValueError('missing actual kernel allocation directives')
    report={}
    for name,source in raw.items():
        changes={}
        def kernel(m):
            old=int(re.search(r'\.amdhsa_next_free_sgpr (\d+)',m[3])[1]);target=limits[m[2]]
            if old!=target:changes[m[2]]=dict(raw_next_free_sgpr=old,matched_next_free_sgpr=target)
            return m[1]+re.sub(r'(\.amdhsa_next_free_sgpr )\d+',lambda x:x[1]+str(target),m[3])+m[4]
        padded=RX.sub(kernel,source)
        for kernel,counts in changes.items():
            pattern=r'(?s)(\.name:\s*'+re.escape(kernel)+r'\s*\n.*?\.sgpr_count:\s*)(\d+)'
            def count(m):return m[1]+str(int(m[2])+counts['matched_next_free_sgpr']-counts['raw_next_free_sgpr'])
            padded,n=re.subn(pattern,count,padded)
            if n!=1:raise ValueError('metadata allocation identity')
        digest=hashlib.sha256(source.encode()).hexdigest();raw_path=out/(name+'.unmatched.'+digest[:12]+'.s')
        if not raw_path.exists():raw_path.write_text(source)
        (out/(name+'.s')).write_text(padded)
        report[name]=dict(changes=changes,raw_assembly=str(raw_path),raw_sha256=digest,matched_sha256=hashlib.sha256(padded.encode()).hexdigest(),instructions_changed=False)
    (out/'g11-allocation-matching.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':print(json.dumps(match(sys.argv[1]),indent=2))
