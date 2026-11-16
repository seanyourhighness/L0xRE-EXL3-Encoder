"""Compare two already-existing head payloads without any new head fitting."""
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from gguf import GGUFReader
from common import read,require,sha,atomic_json
from gguf_patch import export


def compare(plan,candidate):
    from candidates import native_gate
    candidate=Path(candidate);proof=candidate/'head-choice.json'
    if proof.exists():
        old=read(proof);require(sha(old['selected']['model'])==old['selected']['sha256'],'Selected head/body artifact changed')
        for result in old['variants']:
            p=Path(result['model'])
            if result['head']!=old['selected']['head'] and p.exists():
                require(sha(p)==result['sha256'],'Unselected head artifact changed');p.unlink()
        return old['selected']
    original=Path(plan['role_template']);require(sha(original)==plan['role_template_sha256'],'Original head donor changed')
    directory=Path(plan['work'])/'original-head';directory.mkdir(exist_ok=True);patches={}
    t={x.name:x for x in GGUFReader(str(original)).tensors}
    for name in ('output.lowgpu_codes','output.lowgpu_scales','output.lowgpu_zps'):
        p=directory/(name+'.npy')
        if p.exists():require(np.array_equal(np.load(p,mmap_mode='r',allow_pickle=False),t[name].data),'Original head cache changed')
        else:np.save(p,t[name].data)
        patches[name]=str(p)
    side=candidate/'original-head';side.mkdir(exist_ok=True);model=side/'quant.gguf';source=candidate/'quant.gguf'
    if not model.exists():
        result=export(source,model,patches,read(candidate/'native.complete.json')['candidate_sha256'])
        require(result['changed_tensor_count']==3,'Head sidecar changed something besides the existing three head payloads')
        atomic_json(str(model)+'.receipt.json',result)
    child=dict(plan,work=str(side),parent_plan=str(Path(plan['work'])/'plan.json'),candidate_label=candidate.name+'-original-head')
    atomic_json(side/'plan.json',child)
    if not (side/'role-probe').exists():(side/'role-probe').symlink_to(Path(plan['work'])/'role-probe',target_is_directory=True)
    if not (side/'native.complete.json').exists():
        subprocess.run([sys.executable,str(Path(__file__).with_name('qualify.py')),'--plan',str(side/'plan.json'),'--native-only'],check=True)
    results=[]
    for name,root in [('production',candidate),('original',side)]:
        d=read(root/'paired-B02-A2.json');results.append(dict(head=name,model=str(root/'quant.gguf'),sha256=read(root/'native.complete.json')['candidate_sha256'],
            mean=d['candidate_mean'],gate=native_gate(d),native_report_sha256=sha(root/'paired-B02-A2.json')))
    best=min(results,key=lambda r:(not r['gate']['passed'],r['mean']))
    atomic_json(proof,dict(selected=best,variants=results,
        policy='Only pre-existing original and B02-A2 head tensors; no fitting or new codec. Choose using development gates before final holdout.',
        reconstruction='Both variants have identical fitted body tensors and metadata. An unselected variant is exactly reproducible from the selected body plus its pinned donor head.'))
    for result in results:
        if result['head']!=best['head']:
            p=Path(result['model']);require(sha(p)==result['sha256'],'Unselected head artifact changed before reclaim')
            p.unlink()
    return best
