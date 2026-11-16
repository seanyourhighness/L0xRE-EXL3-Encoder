"""New deterministic reasoning holdout, frozen before model selection.

These prompts/answers are independently constructed arithmetic fixtures, not
student-generated traces. Only the original BF16 model supplies probability labels.
They are never used for calibration, sensitivity, factor collection or selection.
"""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from common import read,require,sha,atomic_json


def freeze(plan):
    root=Path(plan['work']);dest=root/'blind-reasoning-reservation.json'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(plan['source'],local_files_only=True,trust_remote_code=False)
    require(bool(tokenizer.chat_template),'Missing original source chat template; do not substitute a template')
    cases=[]
    for i in range(16):
        n=31+7*i;price=137+11*i;fee=19+3*i
        prompt=(f'A workshop makes {n} parts. Each sells for {price} cents. It pays a single {fee}-cent fee for the whole order, '
                f'and then splits the remaining money equally among {i%4+2} people. '
                'Give each person the largest equal whole-cent share and state how many cents are left. Show the integer arithmetic; do not round.')
        net=n*price-fee;people=i%4+2;share,left=divmod(net,people)
        answer=f'Total sales: {n} × {price} = {n*price} cents. After the fee: {n*price} − {fee} = {net} cents. '
        answer+=f'Dividing by {people} gives {share} cents per person, with {left} cents left. Check: {people} × {share} + {left} = {net}.'
        prefix=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        tail=tokenizer.encode(answer,add_special_tokens=False);require(bool(tail),'Empty reasoning answer')
        cases.append(dict(id=f'blind-reasoning-v2-{i:02d}',category='reasoning',prompt=prompt,answer=answer,
                          tokens=list(prefix)+tail,score_from=len(prefix)-1))
    result=dict(cases=cases,tokenizer_sha256=sha(Path(plan['source'])/'tokenizer.json'),
                policy='One frozen development winner only. No calibration/sensitivity/training/tuning on these cases or probabilities.')
    if dest.exists():require(read(dest)==result,'Blind reasoning fixture changed')
    else:atomic_json(dest,result)
    return result


def stage(plan):
    import torch
    from hf_stream import cpu_model,Streaming
    root=Path(plan['work']);reservation=root/'blind-reasoning-reservation.json';data=read(reservation)
    out=root/'blind-reasoning';out.mkdir(exist_ok=True);parts=out/'parts';parts.mkdir(exist_ok=True)
    model=cpu_model(SimpleNamespace(model_dir=plan['source']),[0]);stream=Streaming(model)
    records=[];metadata=dict(format='radiance-kld-top-1',model='Original BF16 '+plan['source_revision'],n_vocab=248320,top=512,rows=0,docs=[])
    try:
        for case in data['cases']:
            part=parts/(case['id']+'.npz');record=part.with_suffix('.json')
            identity=dict(reservation_sha256=sha(reservation),source_revision=plan['source_revision'],implementation_sha256=sha(__file__))
            if part.exists():require(read(record)['identity']==identity and sha(part)==read(record)['sha256'],'Blind label checkpoint changed')
            else:
                seq=np.asarray(case['tokens'],np.int64);begin=case['score_from'];length=len(seq)-1-begin
                with torch.inference_mode():
                    h=model.get_decoder()(input_ids=torch.from_numpy(seq)[None].cuda(),use_cache=False).last_hidden_state[0,begin:-1]
                    logits=np.concatenate([model.get_output_embeddings()(h[j:j+32]).float().cpu().numpy() for j in range(0,len(h),32)])
                require(logits.shape==(length,248320) and np.isfinite(logits).all(),'Incomplete/nonfinite blind teacher logits')
                lp=logits.astype(np.float64);lp-=lp.max(1,keepdims=True);lp-=np.log(np.exp(lp).sum(1,keepdims=True))
                prob=np.exp(lp);idx=np.argsort(-lp,axis=1)[:,:512].astype(np.int32);top=np.take_along_axis(lp,idx,1).astype(np.float32)
                outside=prob.copy();np.put_along_axis(outside,idx,0,axis=1);tail=outside.sum(1)
                rows=np.stack([-lp[np.arange(length),seq[begin+1:]],-(prob*lp).sum(1),tail],1).astype(np.float32)
                require(np.isfinite(rows).all() and (tail>0).all() and (tail<1).all(),'Invalid blind teacher distribution')
                tmp=part.with_suffix('.partial.npz');np.savez(tmp,ids=idx,lp=top,rows=rows);tmp.replace(part)
                atomic_json(record,dict(identity=identity,sha256=sha(part)))
                del h,logits,lp,prob,outside;torch.cuda.empty_cache()
            records.append((case,part))
    finally:stream.close()
    with (out/'top.bin').open('wb') as top,(out/'rows.f32').open('wb') as rows,(out/'tokens.i32').open('wb') as tokens:
        for i,(case,part) in enumerate(records):
            with np.load(part,allow_pickle=False) as d:
                packed=np.empty(d['ids'].shape,dtype=[('id','<i4'),('lp','<f4')]);packed['id']=d['ids'];packed['lp']=d['lp'];packed.tofile(top);d['rows'].tofile(rows)
                count=len(d['rows'])
            np.asarray(case['tokens'],np.int32).tofile(tokens)
            metadata['docs'].append(dict(source='blind-reasoning-v2',tokens=len(case['tokens']),score_from=case['score_from'],row0=metadata['rows']));metadata['rows']+=count
    np.ones(metadata['rows'],np.uint8).tofile(out/'gen.u8');atomic_json(out/'kld.json',metadata)
    atomic_json(out/'selection.json',[dict(original_doc=i,category='reasoning') for i in range(len(records))])
    atomic_json(out/'receipt.json',dict(reservation_sha256=sha(reservation),files={n:sha(out/n) for n in ('top.bin','rows.f32','tokens.i32','gen.u8','kld.json','selection.json')}))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--stage',action='store_true');a=p.parse_args()
    plan=read(a.plan);print(stage(plan) if a.stage else freeze(plan))
