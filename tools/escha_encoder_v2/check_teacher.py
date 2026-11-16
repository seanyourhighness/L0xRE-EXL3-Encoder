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
        candidates=[]
        for i,x in enumerate(selection):
            if x['category']!=category:continue
            doc=reference.docs[i];count=max(0,min(doc['tokens'],max_tokens)-1-doc['score_from'])
            if count and reference.gen[doc['row0']:doc['row0']+count].any():candidates.append(i)
        if candidates:
            i=max(candidates,key=lambda i:min(reference.docs[i]['tokens'],max_tokens));choices.append((i,category))
            # Keep the earlier critical-domain regression cases as well as the
            # longest eligible prefix; do not select around a failed example.
            if category in ('tools','reasoning') and candidates[0]!=i:choices.append((candidates[0],category))
    require({'code','tools','reasoning'}<=set(c for _,c in choices),'Missing critical-domain teacher parity examples')
    return choices


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True)
    p.add_argument('--max-tokens',type=int,default=4096)
    a=p.parse_args();assets=Path(a.assets);source=assets/'source/radiance/tools/score.py'
    spec=importlib.util.spec_from_file_location('retained_reference_scorer',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    refpath=assets/'data/development-expanded-v1';ref=module.Reference(str(refpath));selection=read(refpath/'selection.json')
    model=cpu_model(SimpleNamespace(model_dir=a.model),[0]);stream=Streaming(model);records=[]
    try:
        for doc,category in cases(ref,selection,a.max_tokens):
            entry=ref.docs[doc];n=min(entry['tokens'],a.max_tokens);ids=ref.doc_tokens(doc)[:n].astype(np.int64)
            start=entry['score_from'];end=n-1
            eligible=np.flatnonzero(ref.gen[entry['row0']:entry['row0']+end-start])+start
            require(len(eligible)>0,'No scored assistant tokens in the tested prefix')
            positions=sorted(set(int(x) for x in eligible[np.linspace(0,len(eligible)-1,min(8,len(eligible)),dtype=int)]))
            with torch.inference_mode():
                hidden=model.get_decoder()(input_ids=torch.from_numpy(ids)[None].cuda(),use_cache=False).last_hidden_state
                logits=model.get_output_embeddings()(hidden[:,positions]).float().cpu().numpy()[0]
            values=[]
            for i,pos in enumerate(positions):values.extend(ref.score_rows(doc,pos-start,logits[i:i+1])[0].tolist())
            passed=bool(np.isfinite(values).all() and max(values)<.02 and float(np.mean(values))<.005)
            records.append(dict(document=doc,category=category,prefix_tokens=n,positions=positions,mean_KL=float(np.mean(values)),max_KL=max(values),
                per_position_KL=values,passed=passed,all_positions_match_original_gen_mask=True))
            print('TEACHER_PARITY',records[-1],flush=True)
            atomic_json(str(a.out)+'.diagnostic.json',dict(passed=False,records=records,complete=False,reference_metadata_sha256=sha(refpath/'kld.json')))
        require(all(r['passed'] for r in records),'Original BF16 longer-prefix parity failed; inspect diagnostic before further fitting')
        atomic_json(a.out,dict(passed=True,documents=len(records),positions=sum(len(r['positions']) for r in records),
              records=records,scorer_sha256=sha(source),reference_metadata_sha256=sha(refpath/'kld.json'),
              method='Original BF16, complete prefixes through4096, early/middle/late scored assistant logits under the original gen mask; includes GDN and full-attention blocks'))
    finally:stream.close()


if __name__=='__main__':main()
