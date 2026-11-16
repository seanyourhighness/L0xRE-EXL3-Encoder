"""Reserve new documents before selection; fetch their golden rows only for a winner."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import numpy as np
import requests
from common import atomic_json,read,require,sha

REV='2dcf0061cb683152742b904e2965ce7cd92b2da6'


def freeze(plan):
    root=Path(plan['work']);data=Path(plan['data_source']);assets=Path(plan['assets']);dest=root/'holdout-reservation.json'
    if dest.exists():return read(dest)
    meta=read(data/'public/kld.json');corpus=[json.loads(x) for x in (data/'public.jsonl').read_text().split('\n') if x.strip()]
    require(len(corpus)==len(meta['docs']),'Public holdout metadata/corpus mismatch')
    exclusions=set();receipts={}
    def inspect(value):
        if isinstance(value,list):
            for v in value:inspect(v)
        elif isinstance(value,dict):
            for key,v in value.items():
                if key in ('radiance_public_document_ids','kld_document_ids','original_doc','original_document'):
                    if isinstance(v,int):exclusions.add(v)
                    elif isinstance(v,list):exclusions.update(x for x in v if isinstance(x,int))
                inspect(v)
    files=[assets/'DATA-SPLITS.json',assets/'TASK-MANIFEST.json']
    files+=list((assets/'data').glob('*/selection.json'))+list((assets/'data/radiance').glob('*/selection.json'))
    for p in files:
        if p.exists():inspect(read(p));receipts[str(p)]=sha(p)
    forbidden=set()
    def prompts(value):
        if isinstance(value,list):
            for v in value:prompts(v)
        elif isinstance(value,dict):
            for key,v in value.items():
                if isinstance(v,str) and ('prompt' in key or key in ('question','instruction')):forbidden.add(hashlib.sha256(v.strip().encode()).hexdigest())
                else:prompts(v)
    prompts(read(assets/'TASK-MANIFEST.json'))
    order=sorted(range(len(corpus)),key=lambda i:hashlib.sha256(('V2-final-20261116:'+corpus[i]['prompt']).encode()).hexdigest())
    selected=[];counts={}
    for i in order:
        category=corpus[i]['category']
        if i in exclusions or meta['docs'][i]['tokens']>4096 or counts.get(category,0)>=4:continue
        if hashlib.sha256(corpus[i]['prompt'].strip().encode()).hexdigest() in forbidden:continue
        selected.append(i);counts[category]=counts.get(category,0)+1
        if len(selected)==32:break
    require(len(selected)==32 and {'code','tools'}<=set(counts),
            'Insufficient disjoint final documents: '+str(dict(selected=len(selected),categories=counts,excluded=len(exclusions))))
    result=dict(format='V2-final-reservation',document_ids=selected,categories=counts,dataset_revision=REV,
       metadata_sha256=sha(data/'public/kld.json'),exposure_sources=receipts,prior_exposed_document_ids=sorted(exclusions),
       critical_domain_coverage={c:counts.get(c,0) for c in ('code','tools','reasoning')},
       policy='Reserved before V2 sensitivity/allocation/native results. One selected candidate vs B02-A2; never fit/tune on these outcomes.',
       limitation='Known retained development/final/task overlaps excluded; completeness of every historical exposure is not provable')
    atomic_json(dest,result);return result


def fetch_rows(start,length):
    end=start+length-1;url=f'https://huggingface.co/datasets/StillDeadcode/qwen3.8-27b-quant-challenge/resolve/{REV}/golden/public/top.bin?range={start}-{end}'
    with requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=120) as response:
        response.raise_for_status();require(response.status_code==206 and response.headers['Content-Range'].startswith(f'bytes {start}-{end}/'),'Incorrect golden range')
        raw=response.content;require(len(raw)==length,'Incomplete golden rows');return raw


def stage(plan):
    root=Path(plan['work']);reservation=read(root/'holdout-reservation.json');data=Path(plan['data_source'])/'public';meta=read(data/'kld.json')
    require(sha(data/'kld.json')==reservation['metadata_sha256'],'Reserved holdout source changed')
    corpus=[json.loads(x) for x in (Path(plan['data_source'])/'public.jsonl').read_text().split('\n') if x.strip()]
    out=root/'holdout32';out.mkdir(exist_ok=True);parts=out/'parts';parts.mkdir(exist_ok=True)
    offsets=np.cumsum([0]+[d['tokens'] for d in meta['docs']]);tok=np.memmap(data/'tokens.i32',np.int32,'r')
    rows=np.memmap(data/'rows.f32',np.float32,'r',shape=(meta['rows'],3));mask=np.memmap(data/'gen.u8',np.uint8,'r')
    packed={k:v for k,v in meta.items() if k not in ('docs','rows','corpus_fnv1a64')};packed.update(docs=[],rows=0);selection=[]
    handles={n:(out/(n+'.partial')).open('wb') for n in ('tokens.i32','top.bin','rows.f32','gen.u8')}
    try:
        for i in reservation['document_ids']:
            d=meta['docs'][i];count=d['tokens']-1-d['score_from'];start=d['row0']
            file=parts/(str(i)+'.bin');receipt=file.with_suffix('.json')
            if file.exists():require(sha(file)==read(receipt)['sha256'],'Holdout range cache changed');raw=file.read_bytes()
            else:
                raw=fetch_rows(start*4096,count*4096);tmp=file.with_suffix('.partial');tmp.write_bytes(raw);os.replace(tmp,file);atomic_json(receipt,dict(sha256=sha(file)))
            handles['tokens.i32'].write(tok[offsets[i]:offsets[i+1]].tobytes());handles['top.bin'].write(raw)
            handles['rows.f32'].write(rows[start:start+count].tobytes());handles['gen.u8'].write(mask[start:start+count].tobytes())
            packed['docs'].append(dict(d,row0=packed['rows']));packed['rows']+=count
            selection.append(dict(original_doc=i,category=corpus[i]['category']))
    finally:
        for f in handles.values():f.close()
    for n in handles:os.replace(out/(n+'.partial'),out/n)
    atomic_json(out/'kld.json',packed);atomic_json(out/'selection.json',selection)
    atomic_json(out/'receipt.json',dict(reservation_sha256=sha(root/'holdout-reservation.json'),files={n:sha(out/n) for n in [*handles,'kld.json','selection.json']}))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--stage',action='store_true');a=p.parse_args()
    plan=read(a.plan);print(stage(plan) if a.stage else freeze(plan))
