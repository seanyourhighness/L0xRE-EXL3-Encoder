"""Shared Escha V2 contract. No runtime or upstream numerical code is modified."""
import json
from pathlib import Path
import sys

TREE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TREE/'tools/escha_encoder'))
from adapter import MCG,MODEL_BYTES,UPSTREAM_COMMIT,atomic_json,sha,require,inventory,convert_tiles,validate_scales


def read(path):return json.loads(Path(path).read_text())


def roles(source):
    import re
    from collections import Counter
    matches=re.findall(r'\(ic==(\d+) && oc==(\d+) && K==([23]) && acc==([01])\)\?(\d+):',Path(source).read_text())
    require(len(Counter(matches))==12 and set(Counter(matches).values())=={2},'Runtime role lists disagree')
    table={tuple(map(int,r[:4])):int(r[4]) for r in matches}
    require(len(table)==12 and set(table.values())==set(range(12)),'Unknown runtime dispatch table; audit before allocation')
    return table


def contract(template,bridge_source):
    from gguf import GGUFReader
    rows=inventory(GGUFReader(str(template)));table=roles(bridge_source)
    for r in rows:
        r['allowed_k']=[k for k in (2,3) if (r['ic'],r['oc'],k,int(r['ic']<=6144)) in table]
        require(r['k'] in r['allowed_k'],'Current allocation is not admitted by the pinned runtime')
    return dict(format='l0xre-escha-v2-contract',upstream_commit=UPSTREAM_COMMIT,codebook='mcg',multiplier=MCG,
                byte_ceiling=MODEL_BYTES,template=str(Path(template).resolve()),
                bridge_source_sha256=sha(bridge_source),projections=rows,
                body_bits=sum(r['ic']*r['oc']*r['k'] for r in rows),
                policy='Change body K2/K3 only; preserve vocabulary, metadata, small tensors and bias/corrections')


def validate_recipe(policy,recipe):
    assigned=recipe['tensors'];expected={r['source_key'] for r in policy['projections']}
    require(set(assigned)-{'*'}==expected,'Recipe must cover every coded projection exactly')
    require(assigned.get('*')==16,'Other upstream tensors must stay unquantized')
    bits=0;changes=[]
    for r in policy['projections']:
        k=assigned[r['source_key']]
        require(type(k) is int and k in r['allowed_k'],'Unsupported runtime width: '+r['name'])
        bits+=r['ic']*r['oc']*k
        if k!=r['k']:changes.append(dict(name=r['name'],old=r['k'],new=k,bytes_delta=r['ic']*r['oc']*(k-r['k'])//8))
    require(bits<=policy['body_bits'],'Recipe exceeds coded-body byte budget')
    return dict(body_bits=bits,body_bytes=bits//8,unused_body_bytes=(policy['body_bits']-bits)//8,
                changed_projections=changes,width_counts={str(k):sum(assigned[r['source_key']]==k for r in policy['projections']) for k in (2,3)})
