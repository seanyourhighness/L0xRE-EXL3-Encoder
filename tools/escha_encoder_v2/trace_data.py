"""Prepare higher-volume BF16 teacher calibration without contaminating evaluation."""
import argparse
from collections import defaultdict,deque
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from common import atomic_json,require,sha


def prepare(reference,corpus,public_corpus,target,rows=250,cols=2048,seed=20261116,task_manifest=None):
    reference=Path(reference);target=Path(target)
    metadata=json.loads((reference/'kld.json').read_text());docs=[json.loads(x) for x in Path(corpus).read_text().split('\n') if x.strip()]
    public=[json.loads(x) for x in Path(public_corpus).read_text().split('\n') if x.strip()]
    require(len(docs)==len(metadata['docs']),'Calibration corpus/token metadata mismatch')
    forbidden={hashlib.sha256(x['prompt'].strip().encode()).hexdigest() for x in public}
    if task_manifest:
        def visit(obj):
            if isinstance(obj,list):
                for item in obj:visit(item)
            elif isinstance(obj,dict):
                for key,value in obj.items():
                    if isinstance(value,str) and ('prompt' in key or key in ('question','instruction')):
                        forbidden.add(hashlib.sha256(value.strip().encode()).hexdigest())
                    else:visit(value)
        visit(json.loads(Path(task_manifest).read_text()))
    offsets=np.cumsum([0]+[x['tokens'] for x in metadata['docs']])
    tokens=np.memmap(reference/'tokens.i32',dtype='<i4',mode='r')
    require(len(tokens)==offsets[-1],'Calibration token file length differs')
    buckets=defaultdict(list)
    for i,(doc,entry) in enumerate(zip(docs,metadata['docs'])):
        key=hashlib.sha256(doc['prompt'].strip().encode()).hexdigest()
        if key in forbidden or entry['tokens']<2:continue
        rank=hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()
        buckets[doc['category']].append((rank,i))
    queue={k:deque(i for _,i in sorted(v)) for k,v in sorted(buckets.items())};selection=[]
    while len(selection)<rows and any(queue.values()):
        for category,q in queue.items():
            if q and len(selection)<rows:selection.append((q.popleft(),category))
    require(len(selection)==rows,'Insufficient disjoint calibration documents')
    packed=np.zeros((rows,cols),np.int64);lengths=np.zeros(rows,np.int64);records=[]
    for row,(i,category) in enumerate(selection):
        n=min(cols,int(offsets[i+1]-offsets[i]));value=np.asarray(tokens[offsets[i]:offsets[i]+n]);require((value>=0).all(),'Negative token IDs')
        packed[row,:n]=value;lengths[row]=n;records.append(dict(original_doc=i,category=category,tokens=n,
            prompt_sha256=hashlib.sha256(docs[i]['prompt'].strip().encode()).hexdigest(),truncated=n<metadata['docs'][i]['tokens']))
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        from safetensors.numpy import load_file
        old=load_file(str(target));require(np.array_equal(old['input_ids'],packed) and np.array_equal(old['lengths'],lengths),
                                         'Existing immutable calibration differs')
    else:
        save_file({'input_ids':packed,'lengths':lengths},str(target),metadata={'split':'calibration-only',
            'origin':'retained original-tokenizer prefixes from BF16 reference calibration split; upstream sc_trace packed format',
            'note':'No golden probabilities, public evaluation prompts, or quantized-model generated targets'})
    atomic_json(str(target)+'.receipt.json',dict(rows=rows,cols=cols,tokens=int(lengths.sum()),seed=seed,
        sha256=sha(target),source_sha256={n:sha(p) for n,p in {'metadata':reference/'kld.json','tokens':reference/'tokens.i32',
         'calib_corpus':Path(corpus),'public_corpus':Path(public_corpus)}.items()},selection=records,
        public_prompt_overlap=0,trace_mode='retained-bf16',fresh_sc_trace_generated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--corpus',required=True)
    p.add_argument('--public-corpus',required=True);p.add_argument('--out',required=True)
    p.add_argument('--task-manifest');p.add_argument('--rows',type=int,default=250);p.add_argument('--cols',type=int,default=2048);a=p.parse_args()
    prepare(a.reference,a.corpus,a.public_corpus,a.out,a.rows,a.cols,task_manifest=a.task_manifest)
