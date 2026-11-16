"""Retain earned production vocabulary changes; measure their separate contribution."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from gguf import GGUFReader
from common import read,sha,require,atomic_json,native_runtime
from gguf_patch import export


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=read(a.plan)
    root=Path(plan['work']);v1=Path(plan['v1']);assets=Path(plan['assets'])
    donor=Path(plan['template']);require(sha(donor)==plan['template_sha256'],'Production donor changed')
    base=Path(plan['role_template']);require(sha(base)==plan['role_template_sha256'],'Original geometry donor changed')
    originals={t.name:t for t in GGUFReader(str(base)).tensors};production={t.name:t for t in GGUFReader(str(donor)).tensors}
    differences=[]
    for name,old in originals.items():
        if name.endswith(('.escha_code','.escha_rin','.escha_rout')):continue
        new=production[name]
        require(old.data.shape==new.data.shape and old.data.dtype==new.data.dtype,'Nonbody format change: '+name)
        oldsha=hashlib.sha256(memoryview(old.data).cast('B')).hexdigest();newsha=hashlib.sha256(memoryview(new.data).cast('B')).hexdigest()
        if oldsha!=newsha:differences.append(name)
    expected={'output.lowgpu_codes','output.lowgpu_scales','output.lowgpu_zps'}
    require(set(differences)==expected,'Unexpected nonbody changes; inspect instead of silently adopting: '+str(differences))
    directory=root/'production-head';directory.mkdir(exist_ok=True);patches={}
    for name in sorted(expected):
        file=directory/(name+'.npy');value=production[name].data
        if file.exists():require(np.array_equal(np.load(file,mmap_mode='r',allow_pickle=False),value),'Donor head cache changed')
        else:np.save(file,value)
        patches[name]=str(file)
    model=Path(plan['head_control_model']);model.parent.mkdir(parents=True,exist_ok=True)
    v1_model=v1/'full-escha.gguf';v1_sha=read(str(v1_model)+'.receipt.json')['export']['sha256']
    if not model.exists():
        receipt=export(v1_model,model,patches,v1_sha);atomic_json(str(model)+'.receipt.json',receipt)
    else:
        previous=read(str(model)+'.receipt.json')
        require(previous['source_sha256']==v1_sha and sha(model)==previous['sha256'],'Head control artifact/source changed')
        table={r['name']:r for r in previous['tensors']}
        for name in expected:require(table[name]['after_sha256']==hashlib.sha256(memoryview(production[name].data).cast('B')).hexdigest(),'Head control donor differs')
    runtime=native_runtime(plan);out=root/'head-control-native.json';log=root/'head-control-native.log'
    if not out.exists():
        with log.open('wb') as stream:
            subprocess.run([str(assets/'tools/native_kld'),'--model',str(model),'--ref',str(assets/'data/development-expanded-v1'),
                '--out',str(out),'--ctx','8192','--batch','64','--ubatch','64','--threads','6'],
                env=dict(os.environ,**runtime['environment']),stdout=stream,stderr=subprocess.STDOUT,check=True)
    counts=re.findall(r'L0XRE_FASTPATH_SUMMARY fallthroughs=(\d+)',log.read_text(errors='replace'))
    rows=np.fromfile(str(out)+'.rows',np.float32).reshape(-1,4);mask=np.fromfile(assets/'data/development-expanded-v1/gen.u8',np.uint8)
    require(counts and all(int(v)==0 for v in counts) and len(mask)==len(rows) and np.isfinite(rows).all(),'Head control native execution failed')
    atomic_json(root/'head-control-native.complete.json',dict(complete=True,candidate_sha256=sha(model),
         bytes=model.stat().st_size,masked_KLD=float(rows[mask!=0,0].astype(np.float64).mean()),
         changed_from_V1=sorted(expected),donor_sha256=plan['template_sha256'],
         policy='Existing production head copied verbatim, same3bit codec and dimensions; no new fitting/width/codebook',
         scope='Labeled development control isolating earned output-head changes; promotion requires all normal gates',
         report_sha256=sha(out),rows_sha256=sha(str(out)+'.rows')))


if __name__=='__main__':main()
