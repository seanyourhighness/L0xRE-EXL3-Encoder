"""4090 guardian entry point: projection -> layer -> upstream full conversion -> export.

Every stage stops on failure. Full-model execution is explicitly selected with --full.
Existing L0xRE binaries, libraries, vocabulary and correction tensors remain unchanged.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
from adapter import UPSTREAM_COMMIT, atomic_json, require, sha


@contextmanager
def arguments(argv):
    saved=sys.argv
    sys.argv=[saved[0],*map(str,argv)]
    try:yield
    finally:sys.argv=saved


def call(module,argv):
    with arguments(argv):module.main()


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--work',required=True)
    p.add_argument('--cpu-proof',required=True);p.add_argument('--full',action='store_true')
    a=p.parse_args();assets=Path(a.assets);root=Path(a.work);root.mkdir(parents=True,exist_ok=True)
    lease=assets/'leases/4090/state.json';state=json.loads(lease.read_text())
    require(state['state']=='running' and state['child_pid']==os.getpid(),'Must run as the existing guardian child')
    template=Path('/home/sean/escha-assets/base-models/escha-e3-firstclass-v2.gguf')
    proof=json.loads(Path(a.cpu_proof).read_text())
    require(proof['cpu_compatibility_passed'] and proof['upstream_commit']==UPSTREAM_COMMIT,'CPU compatibility not established')
    require(proof['template_sha256']==sha(template),'CPU proof model differs')
    runtime_file=assets/'source/speed-parity-runtime-sm89-v2.json';runtime=json.loads(runtime_file.read_text())
    require(sha(runtime['environment']['LD_PRELOAD'])==runtime['bridge_sha256'],'Frozen bridge differs')
    require(sha(runtime['environment']['GGML_BACKEND_PATH'])==runtime['CUDA_backend_sha256'],'Frozen CUDA backend differs')
    require(sha(runtime['sealed_settings']['path'])==runtime['sealed_settings']['sha256'],'Frozen settings differ')
    gatefile=root/'compatibility-gates.json'
    gates=dict(template_sha256=proof['template_sha256'],upstream_commit=UPSTREAM_COMMIT,
               cpu_compatibility_passed=True,projection_runtime_passed=False,layer_runtime_passed=False,
               runtime_manifest_sha256=sha(runtime_file),started=time.time())
    if gatefile.exists():
        previous=json.loads(gatefile.read_text())
        for key in ('template_sha256','upstream_commit','runtime_manifest_sha256'):
            require(previous[key]==gates[key],'Pipeline resume identity changed')
        gates=previous
    def save():atomic_json(gatefile,gates)
    save()
    import pilot,export_pilot,workflow
    common=['--template',template,'--inputs',assets/'data/body-block0','--hessians',assets/'data/body-hessians',
            '--native-codec',assets/'tools/native_codec_dual.so','--lease-state',lease,'--output',root/'pilot']
    def qualify(label,model,reference):
        output=root/(label+'.json');log=root/(label+'.log')
        identity=dict(model_sha256=sha(model),runtime_sha256=sha(runtime_file),
                      reference_sha256={f:sha(reference/f) for f in ('kld.json','tokens.i32','top.bin','rows.f32','gen.u8')},
                      executable_sha256=sha(assets/'tools/native_kld'))
        complete=root/(label+'.complete.json')
        if complete.exists():
            old=json.loads(complete.read_text());require(old['identity']==identity,'Runtime checkpoint identity differs')
            require(old['report_sha256']==sha(output) and old['rows_sha256']==sha(str(output)+'.rows'),'Runtime output changed')
            return old
        require(not output.exists() and not log.exists(),'Interrupted runtime evidence exists; inspect it before retry')
        command=[str(assets/'tools/native_kld'),'--model',str(model),'--ref',str(reference),'--out',str(output),
                 '--ctx','8192','--batch','64','--ubatch','64','--threads','6']
        env=dict(os.environ,**runtime['environment'])
        with log.open('wb') as stream:
            subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        fallthroughs=re.findall(r'L0XRE_FASTPATH_SUMMARY fallthroughs=(\d+)',log.read_text(errors='replace'))
        require(bool(fallthroughs) and all(int(v)==0 for v in fallthroughs),'Runtime fell through frozen fast paths')
        rows=np.fromfile(str(output)+'.rows',np.float32).reshape(-1,4)
        masks=np.fromfile(reference/'gen.u8',np.uint8)
        require(len(rows)==len(masks) and np.isfinite(rows).all(),'Nonfinite/incomplete native rows')
        result=dict(identity=identity,passed=True,report_sha256=sha(output),rows_sha256=sha(str(output)+'.rows'),
                    masked_KLD=float(rows[masks!=0,0].astype(np.float64).mean()),positions=int(masks.sum()),
                    numerical_scope='Finite complete inference and zero fallback; quality selection is separate')
        atomic_json(complete,result);return result
    def materialize(model):
        if not model.exists():call(export_pilot,['--pilot',root/'pilot','--template',template,'--output',model])
        receipt=json.loads(Path(str(model)+'.receipt.json').read_text())
        require(sha(model)==receipt['sha256'],'Pilot GGUF changed')
    if not gates['projection_runtime_passed']:
        call(pilot,common+['--projection','blk.0.attn_gate'])
        model=root/'projection.gguf';materialize(model)
        gates['projection_runtime']=qualify('projection-runtime',model,assets/'data/smoke8')
        gates['projection_runtime_passed']=True;save()
    if not gates['layer_runtime_passed']:
        call(pilot,common+['--layer','0'])
        model=root/'layer0.gguf';materialize(model)
        gates['layer_runtime']=qualify('layer0-runtime',model,assets/'data/smoke8')
        gates['layer_runtime_passed']=True;save()
    if not a.full:return
    work=root/'full';manifest=work/'manifest.json'
    if not manifest.exists():
        workflow.prepare(argparse.Namespace(work=work,template=template,source=assets/'data/teacher',
                         calibration=root.parent/'calibration64.safetensors'))
    if not (work/'conversion.complete.json').exists():
        workflow.run_conversion(argparse.Namespace(work=work,lease_state=lease,gates=gatefile,
                                resume=(work/'conversion/ckpt/job.json').exists()))
    completed=json.loads((work/'conversion.complete.json').read_text())
    require(completed['manifest_sha256']==sha(manifest),'Conversion manifest changed')
    for path,digest in completed['files'].items():
        require(sha(work/path)==digest,'Converted checkpoint changed')
    output=root/'full-escha.gguf'
    if not output.exists():
        workflow.export_model(argparse.Namespace(work=work,checkpoint=work/'converted',output=output,
                                               projection=None,layer=None))
    result=qualify('full-development116',output,assets/'data/development-expanded-v1')
    baseline=assets/'evidence/stacking-loop/B02/A2/full116.json'
    comparison=root/'paired-development116.json'
    if not comparison.exists():
        subprocess.run([sys.executable,str(assets/'tools/paired_quality_subset.py'),
                        '--baseline',str(baseline),'--candidate',str(root/'full-development116.json'),
                        '--ref',str(assets/'data/development-expanded-v1'),
                        '--candidate-sha256',result['identity']['model_sha256'],'--out',str(comparison)],check=True)
    gates.update(full_model_exported=True,full_development=result,
                 baseline_KLD=.09813687512473697,below_baseline=result['masked_KLD']<.09813687512473697,
                 paired_comparison=str(comparison),paired_comparison_sha256=sha(comparison),
                 coding_qualified=False,dflash_qualified=False,release_qualified=False)
    save()


if __name__=='__main__':main()
