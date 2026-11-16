"""Freeze independent, category-stratified SC replicates and shared shaping rows."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file,save_file
from common import atomic_json,read,require,sha


def select(records,n,critical_min,excluded,seed):
    buckets={}
    for i,r in enumerate(records):
        if i not in excluded:buckets.setdefault(r['category'],[]).append(i)
    for category,indices in buckets.items():indices.sort(key=lambda i:hashlib.sha256(f'{seed}:{records[i]["prompt_sha256"]}'.encode()).hexdigest())
    chosen=[]
    require({'code','tools','reasoning'}<=set(buckets),'Critical categories missing')
    for category in sorted(buckets):
        count=critical_min if category in ('code','tools','reasoning') else 1
        require(len(buckets[category])>=count,'Not enough independent '+category+' documents')
        chosen.extend(buckets[category][:count]);buckets[category]=buckets[category][count:]
    require(len(chosen)<=n,'Requested sample cannot cover all categories and critical-domain minima')
    while len(chosen)<n and any(buckets.values()):
        for category in sorted(buckets):
            if buckets[category] and len(chosen)<n:chosen.append(buckets[category].pop(0))
    require(len(chosen)==n,'Insufficient independent documents')
    return chosen


def prepare(trace,out,rows=24,hrows=64,critical_min=4,seed=20261116):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);records=read(str(trace)+'.receipt.json')['selection'];data=load_file(str(trace))
    a=select(records,rows,critical_min,set(),seed)
    b=select(records,rows,critical_min,set(a),seed+1)
    rest=[i for i in range(len(records)) if i not in set(a+b)]
    rest.sort(key=lambda i:hashlib.sha256(f'{seed}:H:{records[i]["prompt_sha256"]}'.encode()).hexdigest());h=rest[:hrows]
    require(len(h)==hrows and not(set(a)&set(b) or set(a+b)&set(h)),'Replicate/Hessian partition overlap')
    receipt=dict(trace_sha256=sha(trace),measurement_rows=rows,shaping_rows=hrows,critical_min=critical_min,
                 independent_eval_documents=True,shared_shaping_rows=True,replicates={})
    for name,indices in [('A',a),('B',b)]:
        all_ids=indices+h;target=out/(name+'.safetensors')
        arrays={k:v[all_ids].copy() for k,v in data.items()}
        if target.exists():
            old=load_file(str(target));require(all(np.array_equal(old[k],v) for k,v in arrays.items()),'Replicate changed')
        else:save_file(arrays,str(target),metadata={'role':'Independent SC evaluation subset followed by shared shaping subset'})
        receipt['replicates'][name]=dict(file=str(target),sha256=sha(target),
            measurement_ids=[records[i]['original_doc'] for i in indices],shaping_ids=[records[i]['original_doc'] for i in h],
            measurement_categories=dict(Counter(records[i]['category'] for i in indices)),
            measured_tokens_at_1024=int(np.minimum(arrays['lengths'][:rows],1024).sum()))
    atomic_json(out/'selection.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace',required=True);p.add_argument('--out',required=True)
    p.add_argument('--rows',type=int,default=24);p.add_argument('--h-rows',type=int,default=64)
    p.add_argument('--critical-min',type=int,default=4);a=p.parse_args();prepare(a.trace,a.out,a.rows,a.h_rows,a.critical_min)
