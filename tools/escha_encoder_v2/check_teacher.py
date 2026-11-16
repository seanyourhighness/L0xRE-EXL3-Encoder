"""Prove original-source forward fidelity before collecting Fisher factors."""
import argparse
import importlib.util
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from common import atomic_json,read,require,sha
from hf_stream import cpu_model,Streaming


def cases(reference,selection,max_tokens):
    choices=[]
    for category in ('code','tools','reasoning','agentic','web','multilingual'):
        candidates=[i for i,x in enumerate(selection) if x['category']==category and reference.docs[i]['score_from']<max_tokens-1]
        if candidates:
            i=max(candidates,key=lambda i:min(reference.docs[i]['tokens'],max_tokens));choices.append((i,category))
    require({'code','tools','reasoning'}<=set(c for _,c in choices),'Missing critical-domain teacher parity examples')
    return choices


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p.add_argument('--max-tokens',type=int,default=2048)
    a=p.parse_args();assets=Path(a.assets);source=assets/'source/radiance/tools/score.py'
    spec=importlib.util.spec_from_file_location('retained_reference_scorer',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    refpath=assets/'data/development-expanded-v1';ref=module.Reference(str(refpath));selection=read(refpath/'selection.json')
    model=cpu_model(SimpleNamespace(model_dir=a.model),[0]);stream=Streaming(model);records=[]
    try:
        for doc,category in cases(ref,selection,a.max_tokens):
            entry=ref.docs[doc];n=min(entry['tokens'],a.max_tokens);ids=ref.doc_tokens(doc)[:n].astype(np.int64)
            start=entry['score_from'];end=n-1
            positions=sorted(set([start,min(start+1,end-1),max(start,(start+end)//2),max(start,end-8),end-1]))
            with torch.inference_mode():
                hidden=model.get_decoder()(input_ids=torch.from_numpy(ids)[None].cuda(),use_cache=False).last_hidden_state
                logits=model.get_output_embeddings()(hidden[:,positions]).float().cpu().numpy()[0]
            values=[]
            for i,pos in enumerate(positions):values.extend(ref.score_rows(doc,pos-start,logits[i:i+1])[0].tolist())
            require(np.isfinite(values).all() and max(values)<.02 and float(np.mean(values))<.005,
                    'Original BF16 longer-prefix parity failed: '+category)
            records.append(dict(document=doc,category=category,prefix_tokens=n,positions=positions,mean_KL=float(np.mean(values)),max_KL=max(values)))
            print('TEACHER_PARITY',records[-1],flush=True)
        atomic_json(a.out,dict(passed=True,documents=len(records),positions=sum(len(r['positions']) for r in records),
              records=records,scorer_sha256=sha(source),reference_metadata_sha256=sha(refpath/'kld.json'),
              method='Original BF16, complete prefixes through2048, early/middle/late logits; includes GDN and full-attention blocks'))
    finally:stream.close()


if __name__=='__main__':main()
