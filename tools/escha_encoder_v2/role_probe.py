"""Admission fixture covering every changed-width role and all four FFN pairs.

Silenced projections test only format/dispatch; this is never a quality candidate
or an initialization for the real BF16 encode.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from gguf import GGUFReader
from common import read,require,atomic_json,sha,native_runtime
from structural_export import export


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=read(a.plan)
    root=Path(plan['work'])/'role-probe';root.mkdir(parents=True,exist_ok=True);assets=Path(plan['assets'])
    reader=GGUFReader(plan['template']);t={x.name:x for x in reader.tensors};rows={r['name']:r for r in plan['policy']['projections']}
    widths={'blk.0.attn_gate':3,'blk.0.ssm_out':3,'blk.0.ffn_gate':3,'blk.0.ffn_up':2,'blk.0.ffn_down':2,
            'blk.3.attn_k':3,'blk.3.attn_v':3,'blk.1.ffn_up':2,'blk.2.ffn_gate':3}
    patches={}
    for name,k in widths.items():
        r=rows[name];require(k in r['allowed_k'],'Probe includes an unsupported role')
        values={'escha_code':np.zeros((r['ic']//16,r['oc']//16,16*k),np.int16),
                'escha_rin':(t[name+'.escha_rin'].data.astype(np.float32)*.01).astype(np.float16),
                'escha_rout':t[name+'.escha_rout'].data.copy()}
        for suffix,v in values.items():
            file=root/(name+'.'+suffix+'.npy');np.save(file,v);patches[name+'.'+suffix]=str(file)
    model=root/'dispatch-only.gguf'
    if not model.exists():export(plan['template'],model,patches,plan['template_sha256'])
    spec=importlib.util.spec_from_file_location('frozen_runtime_admission',assets/'tools/runtime-profile-v1/audit_fast_path_v1.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit);result=audit.audit(model,plan['bridge_source'])
    require(result['passed'] and all(result['role_counts'][str(i)]>0 for i in range(12)),'Not all runtime roles covered')
    require(set(result['FFN_pair_counts'])=={'(2, 2)','(2, 3)','(3, 2)','(3, 3)'},'Missing FFN width pairing')
    runtime=native_runtime(plan);out=root/'native.json';log=root/'native.log'
    with log.open('wb') as stream:
        subprocess.run([str(assets/'tools/native_kld'),'--model',str(model),'--ref',str(assets/'data/smoke8'),
            '--out',str(out),'--ctx','8192','--batch','64','--ubatch','64','--threads','6'],
            env=dict(os.environ,**runtime['environment']),stdout=stream,stderr=subprocess.STDOUT,check=True)
    counts=re.findall(r'L0XRE_FASTPATH_SUMMARY fallthroughs=(\d+)',log.read_text(errors='replace'))
    values=np.fromfile(str(out)+'.rows',np.float32)
    require(counts and all(int(x)==0 for x in counts) and len(values)>0 and np.isfinite(values).all(),'Runtime admission fixture failed')
    atomic_json(root/'complete.json',dict(passed=True,CPU_admission=result,native_report_sha256=sha(out),
          scope='All12roles/all4FFNpairs; finite full inference and zero fallback. Silenced fixture is NOT a quality model.',
          fixture_sha256=sha(model)))


if __name__=='__main__':main()
