"""V2 native quality vs B02-A2 and V1, followed by the unchanged paired suite."""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from common import TREE,read,require,sha,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=read(a.plan)
    root=Path(plan['work']);assets=Path(plan['assets']);v1=Path(plan['v1']);model=root/'quant.gguf'
    require(sha(model)==read(str(model)+'.receipt.json')['sha256'],'Exported candidate changed')
    from safetensors import safe_open
    require(read(root/'role-probe/complete.json')['passed'],'New-width runtime admission missing')
    rtfile=assets/'source/speed-parity-runtime-sm89-v2.json';runtime=read(rtfile)
    require(sha(runtime['environment']['LD_PRELOAD'])==runtime['bridge_sha256'],'Runtime bridge changed')
    require(sha(runtime['environment']['GGML_BACKEND_PATH'])==runtime['CUDA_backend_sha256'],'Runtime backend changed')
    ref=assets/'data/development-expanded-v1';report=root/'development116.json'
    if not report.exists():
        with (root/'development116.log').open('wb') as out:
            subprocess.run([str(assets/'tools/native_kld'),'--model',str(model),'--ref',str(ref),'--out',str(report),
                '--ctx','8192','--batch','64','--ubatch','64','--threads','6'],env=dict(os.environ,**runtime['environment']),stdout=out,stderr=subprocess.STDOUT,check=True)
    counts=re.findall(r'L0XRE_FASTPATH_SUMMARY fallthroughs=(\d+)',(root/'development116.log').read_text(errors='replace'))
    rows=np.fromfile(str(report)+'.rows',np.float32).reshape(-1,4);mask=np.fromfile(ref/'gen.u8',np.uint8)
    require(counts and all(int(v)==0 for v in counts) and len(mask)==len(rows) and np.isfinite(rows).all(),'Native full-model quality execution failed')
    mean=float(rows[mask!=0,0].astype(np.float64).mean());comparisons={}
    for label,baseline in [('B02-A2',assets/'evidence/stacking-loop/B02/A2/full116.json'),('V1',v1/'full-development116.json')]:
        out=root/('paired-'+label+'.json')
        if not out.exists():
            subprocess.run([sys.executable,str(assets/'tools/paired_quality_subset.py'),'--baseline',str(baseline),
                '--candidate',str(report),'--ref',str(ref),'--candidate-sha256',sha(model),'--out',str(out)],check=True)
        comparisons[label]=dict(path=str(out),sha256=sha(out))
    # Reuse the exact V1 qualification implementation/fixtures, with a separate
    # candidate/cache namespace. Hard links add no second model-sized copy.
    alias=root/'full-escha.gguf'
    if not alias.exists():os.link(model,alias)
    catalog=root.parent/'benchlocal099'
    if not catalog.exists():catalog.symlink_to(v1.parent/'benchlocal099',target_is_directory=True)
    v1g=read(v1/'compatibility-gates.json');g=dict(v1g,full_model_exported=True,layer_runtime_passed=True,
         projection_runtime_passed=True,qualification_complete=False,coding_qualified=False,dflash_qualified=False,
         full_development=dict(identity=dict(model_sha256=sha(model))),below_baseline=mean<.09813687512473697,
         candidate='EXL3-MCG-SC-YaQA-V2',masked_KLD=mean,comparisons=comparisons)
    atomic_json(root/'compatibility-gates.json',g)
    source=TREE/'tools/escha_encoder/qualify.py';spec=importlib.util.spec_from_file_location('unchanged_paired_qualification',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=sys.argv;sys.argv=[str(source),'--assets',str(assets),'--work',str(root)]
    try:module.main()
    finally:sys.argv=old
    result=read(root/'compatibility-gates.json')
    atomic_json(root/'qualification.complete.json',dict(complete=True,candidate_sha256=sha(model),masked_KLD=mean,
          comparisons=comparisons,coding_qualified=result['coding_qualified'],dflash_qualified=result['dflash_qualified'],
          release_qualified=False,promoted=False,development_data_not_untouched_holdout=True))


if __name__=='__main__':main()
