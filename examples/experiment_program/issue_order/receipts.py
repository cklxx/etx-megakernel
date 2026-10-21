"""Strict native artifact validation, including every untrimmed raw sample.

This postprocessor is separate from native compilation identity. Every parent
session and winner integration must use it before accepting a micro verdict.
"""
import argparse
import json
import math
from pathlib import Path
import re
from .evidence import require
from examples.experiment_program.protocol import resolve_path
from .__main__ import parse_pair_means as parse_pairs

SAMPLE=re.compile(r'^IO_SAMPLE arm=(control|candidate) device_ns=(\S+) host_ns=(\S+) reset_ns=(\S+) epoch=(\d+)$')
PAIR=re.compile(r'^IO_PAIR ordinal=(\d+) control_ns=(\S+) candidate_ns=(\S+) control_host_ns=(\S+) candidate_host_ns=(\S+) samples=(\d+) layer=(\d+) position=(\d+)$')
CHECK=re.compile(r'^IO_CHECK OK record=(G0[678]) layer=(\d+) position=(\d+) endpoints=(\d+) epochs=(\d+)$')

def validate_native_log(text,record=None,expected_warmup=None,expected_samples=None):
    decision=parse_pairs(text);warmups=[];pairs={};current=None;epochs=[];check=None;endpoint=None
    for line in text.splitlines():
        if line.startswith('STAGE warmup='):
            require(current is None and not pairs,'warmup after measured population');warmups.append(int(line.split('=')[1]))
        elif line.startswith('STAGE pair='):
            match=re.fullmatch(r'STAGE pair=(\d+) first=(\d+)',line);require(match is not None,'pair stage syntax')
            require(current is None,'unfinished native pair');ordinal,first=map(int,match.groups());require(ordinal in range(5) and ordinal not in pairs and first==ordinal%2,'pair ordinal/order changed')
            current={'ordinal':ordinal,'first':first,'rows':[]}
        elif line.startswith('IO_SAMPLE'):
            m=SAMPLE.fullmatch(line);require(m is not None and current is not None,'orphan/malformed raw sample')
            arm,device,host,reset,epoch=m.groups();values=list(map(float,(device,host,reset)));epoch=int(epoch)
            require(all(math.isfinite(x) for x in values) and values[0]>0 and values[1]>0 and values[2]>=0,'invalid raw timing')
            expected=('control','candidate')[(current['first']+len(current['rows']))%2]
            require(arm==expected,'raw AB/BA order mismatch')
            current['rows'].append((arm,*values));epochs.append(epoch)
        elif line.startswith('IO_PAIR'):
            m=PAIR.fullmatch(line);require(m is not None and current is not None,'orphan/malformed pair summary')
            ordinal=int(m[1]);samples=int(m[6]);require(ordinal==current['ordinal'] and 1<=samples<=1000 and len(current['rows'])==2*samples,'truncated/extra raw pair samples')
            if expected_samples is not None:require(samples==expected_samples,'declared sample count changed')
            require(endpoint is None or endpoint==m.groups()[6:8],'mixed native layer/position');endpoint=m.groups()[6:8]
            for arm,device,host in (('control',float(m[2]),float(m[4])),('candidate',float(m[3]),float(m[5]))):
                rows=[row for row in current['rows'] if row[0]==arm];require(len(rows)==samples,'incomplete arm samples')
                require(math.isclose(sum(row[1] for row in rows)/samples,device,rel_tol=1e-12,abs_tol=1e-6),'device mean/raw mismatch')
                require(math.isclose(sum(row[2] for row in rows)/samples,host,rel_tol=1e-12,abs_tol=1e-6),'host mean/raw mismatch')
            pairs[ordinal]=current;current=None
        elif line.startswith('IO_CHECK'):
            m=CHECK.fullmatch(line);require(m is not None and check is None,'malformed/duplicate correctness endpoint');check=m.groups()
    require(current is None and set(pairs)==set(range(5)),'incomplete native pair population')
    require(warmups==list(range(len(warmups))) and 1<=len(warmups)<=100,'missing/duplicate/reordered warmup population')
    if expected_warmup is not None:require(len(warmups)==expected_warmup,'declared warmup changed')
    require(check is not None and (record is None or check[0]==record),'native record mismatch')
    require(tuple(check[1:3])==tuple(endpoint),'native endpoint not bound')
    require(epochs==list(range(2*len(warmups)+1,2*len(warmups)+1+len(epochs))) and int(check[4])==epochs[-1],'missing/duplicate/stale native sample epoch')
    return {'decision':decision,'record':check[0],'layer':int(check[1]),'position':int(check[2]),'warmup_rounds':len(warmups),'raw_samples':len(epochs),'pair_ordinals':list(pairs),'complete':True}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);p.add_argument('--record',choices=['G06','G07','G08']);p.add_argument('--warmup',type=int);p.add_argument('--samples',type=int);a=p.parse_args();print(json.dumps(validate_native_log(resolve_path(a.log).read_text(),a.record,a.warmup,a.samples),indent=2))
if __name__=='__main__':main()
