"""V2 automation. Run as a queued child only after V1's complete qualification.

SC measurement/optimization and YaQA math are the unmodified pinned upstream tools.
V2 controls data provenance, device residency, byte budget, export and admission.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import TREE,UPSTREAM_COMMIT,MCG,MODEL_BYTES,atomic_json,read,require,sha,contract,validate_recipe


def prepare(a):
    root=Path(a.work).resolve();root.mkdir(parents=True,exist_ok=True)
    policy=contract(a.template,a.bridge_source)
    source=Path(a.assets)/'data/teacher'
    data=dict(format='l0xre-escha-v2',upstream_commit=UPSTREAM_COMMIT,work=str(root),assets=str(Path(a.assets).resolve()),
              v1=str(Path(a.v1).resolve()),source=str(source.resolve()),source_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
              extension=str(Path(a.v1).resolve().parent/'torch-extensions/exllamav3_ext/exllamav3_ext.so'),
              extension_sha256=sha(Path(a.v1).resolve().parent/'torch-extensions/exllamav3_ext/exllamav3_ext.so'),
              template=policy['template'],template_sha256='b0849250c633aa93853bf119a877dbdafdd7b1a4ebb672bd6b6a1439906f3543',
              bridge_source=str(Path(a.bridge_source).resolve()),policy=policy,
              calibration_rows=250,calibration_cols=2048,measurement_rows=24,measurement_cols=1024,
              measurement_critical_min=4,measurement_draws=2,measurement_replicates=['A','B'],measurement_memory_gb=24,rfn_scales=[2.5,1.96,1.0,.5],
              quality_review='SOL-HANDOFF/REVIEW2026-11-16: bracket demotions, independent stratified replicates, exact discrete audit, fixed-map control, broad teacher parity, native holdout',
              hessian_rows=64,yaqa_rows=250,yaqa_cols=2048,yaqa_chunk=2,yaqa_sides='out',yaqa_samples=1,
              cpu_memory_max='96G',seed=20261116,pack_workers=3,
              trace_model=a.trace_model,
              validation_pins=dict(runtime_manifest=sha(Path(a.assets)/'source/speed-parity-runtime-sm89-v2.json'),
                  native_scorer=sha(Path(a.assets)/'tools/native_kld'),
                  baseline_rows=sha(Path(a.assets)/'evidence/stacking-loop/B02/A2/full116.json.rows')),
              data_source=str((root.parent/'trace-source').resolve()),
              data_source_pins_sha256=sha(root.parent/'trace-source/pins.json'),
              holdout_source_pins_sha256=sha(root.parent/'trace-source/holdout-pins.json'),
              factor_stores=[str(root/'factor-chunks'),str(Path.home()/('work/l0xre-exl3-v2-factors-'+root.name))],
              implementation={str(p.relative_to(TREE)):sha(p) for folder in (Path(__file__).parent,TREE/'tools/escha_encoder') for p in folder.glob('*.py')},
              upstream_tools={n:sha(TREE/n) for n in ('sc_trace.py','sc_measure.py','sc_optimize.py','sc_rfn_probe.py','util/yaqa_hessians.py',
                  'exllamav3/modules/quant/exl3_lib/quantize.py','exllamav3/exllamav3_ext/quant/codebook.cuh',
                  'exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh','exllamav3/exllamav3_ext/quant/quantize_tiles_optimized.cuh')},
              trace_default='Retained original-tokenizer prefixes from BF16 reference calibration split, in sc_trace format; no V1 low-bit self-generation',
              no_new_runtime_kernels=True,quality_not_guaranteed=True)
    path=root/'plan.json'
    if path.exists():require(read(path)==data,'Plan changed; use a new work directory')
    else:atomic_json(path,data)
    atomic_json(root/'contract.json',policy)
    print(json.dumps(dict(plan=str(path),flexible=sum(len(r['allowed_k'])==2 for r in policy['projections']),
          fixed=sum(len(r['allowed_k'])==1 for r in policy['projections']),ceiling=MODEL_BYTES,stages=['trace','BF16-and-autograd-parity',
          'new-width-runtime','YaQA-pilot','two-sided-MCG-pilot','rfn','shaped-sensitivity','budgeted-upstream-optimizer',
          'YaQA-full-Hout','two-sided-LDLQ','variable-width-GGUF','native-KLD','coding-DFlash']),indent=2))


class Stages:
    def __init__(self,root):self.root=root;self.identity=sha(root/'plan.json');self.python=sys.executable
    def run(self,name,command,outputs,timeout=24*3600):
        folder=self.root/'stages'/name;folder.mkdir(parents=True,exist_ok=True);receipt=folder/'complete.json'
        identity=dict(plan_sha256=self.identity,argv=list(map(str,command)),inputs={})
        output_set={str(Path(x)) for x in outputs}
        for arg in map(str,command):
            if arg.startswith('/') and Path(arg).is_file() and arg not in output_set:identity['inputs'][arg]=sha(arg)
        if receipt.exists():
            d=read(receipt);require(d['identity']==identity,'Stage identity changed: '+name)
            bad=[p for p,h in d['outputs'].items() if not Path(p).exists() or sha(p)!=h]
            if not bad:print('V2_REUSED',name,flush=True);return d
            archive=folder/('corrupt-'+str(time.time_ns()));archive.mkdir()
            for index,p in enumerate(d['outputs']):
                file=Path(p);owned=[self.root,*map(Path,read(self.root/'plan.json')['factor_stores'])]
                require(any(file.is_relative_to(x) for x in owned),'Cannot recover external source files')
                if file.exists():
                    import shutil
                    shutil.move(str(file),str(archive/(str(index)+'-'+file.name)))
            os.rename(receipt,archive/'complete.json');atomic_json(archive/'reason.json',dict(damaged=bad))
        index=len(list(folder.glob('attempt-*.json')))+1;attempt=folder/f'attempt-{index:03d}.json';log=folder/f'attempt-{index:03d}.log'
        record=dict(identity=identity,started=time.time(),complete=False)
        atomic_json(attempt,record);print('V2_STAGE',name,flush=True)
        try:
            with log.open('wb') as stream:
                subprocess.run(list(map(str,command)),cwd=TREE,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)
            require(all(Path(p).exists() for p in outputs),'Stage did not produce its required artifacts: '+name)
            all_outputs=list(map(Path,outputs))
            for p in outputs:
                if Path(p).name=='collector.json':
                    all_outputs.extend(Path(p).parent/n for n in read(p)['files'])
                if Path(p).name=='quantization_config.json':all_outputs.extend(Path(p).parent.glob('*.safetensors'))
            parents=set()
            for p in all_outputs:
                with p.open('rb') as f:os.fsync(f.fileno())
                parents.add(p.parent)
            for parent in parents:
                fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            record.update(complete=True,outputs={str(p):sha(p) for p in all_outputs})
            record['ended']=time.time();record['seconds']=record['ended']-record['started']
            atomic_json(receipt,record);return record
        except Exception as error:
            record['error']=type(error).__name__+': '+str(error)
            raise
        finally:
            record['ended']=time.time();atomic_json(attempt,record)


def v1_gate(plan):
    v1=Path(plan['v1']);g=read(v1/'compatibility-gates.json')
    require(g.get('full_model_exported') is True and g.get('qualification_complete') is True,
            'V1 full export, KLD and coding/DFlash measurements must finish before V2 GPU work')
    require((v1/'full/conversion.complete.json').exists() and (v1/'full-escha.gguf.receipt.json').exists(),'V1 completion evidence missing')
    require(g.get('layer_runtime_passed') and g.get('projection_runtime_passed'),'V1 numerical compatibility failed')
    # Quality result may be negative: V2 is an independent candidate, not a promotion.
    return g


def run(a):
    root=Path(a.work).resolve();plan=read(root/'plan.json');policy=read(root/'contract.json')
    require(plan['policy']==policy,'Policy differs from frozen plan')
    for name,h in {**plan['implementation'],**plan['upstream_tools']}.items():require(sha(TREE/name)==h,'Source pin changed: '+name)
    require(sha(plan['extension'])==plan['extension_sha256'],'Pinned numerical extension changed')
    state=read(Path(plan['assets'])/'leases/4090/state.json')
    require(state['state']=='running' and state.get('child_pid')==os.getpid(),'Must execute as the guardian child')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==state['gpu'],'Guardian GPU mismatch')
    v1_gate(plan)
    completion=read(Path(plan['v1'])/'full/conversion.complete.json')
    for file,h in completion['files'].items():require(sha(Path(plan['v1'])/'full'/file)==h,'V1 anchor checkpoint changed')
    download=read(Path(plan['source'])/'download-state.json')
    require(download['state']=='complete' and download['revision']==plan['source_revision'],'Source cache revision changed')
    cache=root/'source-verified.json';verified=read(cache) if cache.exists() else {};current={}
    for item in download['files']:
        file=Path(plan['source'])/item['file'];st=file.stat()
        signature=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
        previous=verified.get(item['file'])
        if not previous or previous.get('stat')!=signature or previous.get('sha256')!=item['sha256']:
            require(sha(file)==item['sha256'],'Original BF16 source differs: '+item['file'])
        current[item['file']]=dict(stat=signature,sha256=item['sha256'])
    atomic_json(cache,current)
    require(sha(plan['template'])==plan['template_sha256'],'Template changed')
    require(sha(plan['bridge_source'])==policy['bridge_source_sha256'],'Runtime dispatch source changed')
    import shutil
    # Split the ~100GB packed output factors between two owned stores. Leave
    # room for live OS files and GPU-conversion/export scratch without deleting V1.
    demands={};paths={};seen=set();occupied=0
    for pattern in ('conversion/qtensors/*.safetensors','converted/*.safetensors','export-checkpoints/*/*.npy',
                    'conversion/ckpt*/*.safetensors','*.gguf','role-probe/*.gguf','candidates/*/*.gguf',
                    'candidates/*/conversion/qtensors/*.safetensors','candidates/*/converted/*.safetensors',
                    'candidates/*/export-checkpoints/*/*.npy','candidates/*/conversion/ckpt*/*.safetensors'):
        for f in root.glob(pattern):
            st=f.stat();key=(st.st_dev,st.st_ino)
            if key not in seen:occupied+=st.st_size;seen.add(key)
    device=root.stat().st_dev;demands[device]=max(80*1024**3-occupied,16*1024**3);paths[device]=root
    for index,store_name in enumerate(plan['factor_stores']):
        store=Path(store_name);store.mkdir(parents=True,exist_ok=True);device=store.stat().st_dev;paths[device]=store
        owner=store/'OWNER.json'
        if owner.exists():require(read(owner)['work']==str(root),'Factor store belongs to another run')
        else:
            require(not any(store.iterdir()),'Refuse an unowned nonempty factor store')
            atomic_json(owner,dict(work=str(root)))
        expected=sum((r['oc']*(r['oc']+1)//2+r['oc'])*4 for r in policy['projections'] if (r['layer']//plan['yaqa_chunk'])%len(plan['factor_stores'])==index)
        present=sum(f.stat().st_size for f in store.glob('*/*.safetensors'))
        demands[device]=demands.get(device,20*1024**3)+max(expected-present,0)
    for device,needed in demands.items():require(shutil.disk_usage(paths[device]).free>needed,'Insufficient factor/workspace capacity: '+str(paths[device]))
    # Own reservation only; persist no system-wide memory/CPU policy change.
    subprocess.run(['systemctl','--user','set-property','--runtime','l0xre-exl3-4090-reservation.service','MemoryMax=96G'],check=True)
    s=Stages(root);python=sys.executable;here=Path(__file__).parent;assets=Path(plan['assets']);v1=Path(plan['v1']);source=Path(plan['source'])
    atomic_json(root/'v1-prerequisite.json',dict(passed=True,v1_gate_sha256=sha(v1/'compatibility-gates.json')))
    from holdout import freeze
    freeze(plan)
    from blind_reasoning import freeze as freeze_reasoning
    freeze_reasoning(plan)
    trace=root/'calibration.safetensors'
    data=Path(plan['data_source']);pins=read(data/'pins.json')
    require(sha(data/'pins.json')==plan['data_source_pins_sha256'],'Calibration provenance changed')
    for name,h in pins['files'].items():require(sha(data/name)==h,'Calibration source changed: '+name)
    require(sha(data/'holdout-pins.json')==plan['holdout_source_pins_sha256'],'Holdout source provenance changed')
    for name,h in read(data/'holdout-pins.json')['files'].items():require(sha(data/name)==h,'Holdout token metadata changed: '+name)
    if plan['trace_model']:
        teacher=Path(plan['trace_model']);cfg=read(teacher/'quantization_config.json')
        require(cfg['quant_method']=='exl3' and cfg['bits']>=6,'Fresh self-trace needs a verified high-precision EXL3 teacher, not V1')
        require(sha(teacher/'tokenizer.json')==sha(source/'tokenizer.json'),'Fresh trace teacher tokenizer differs from the source')
        s.run('trace',[python,TREE/'sc_trace.py','-m',teacher,'-o',root/'trace.json','-co',trace,'-cr','250','-cc','2048',
              '--seed',str(plan['seed'])],[trace,root/'trace.json'])
    else:
        s.run('trace',[python,here/'trace_data.py','--reference',data/'calib','--corpus',data/'calib.jsonl',
              '--public-corpus',data/'public.jsonl','--task-manifest',assets/'TASK-MANIFEST.json','--out',trace,
              '--rows',str(plan['calibration_rows']),'--cols',str(plan['calibration_cols'])], [trace,Path(str(trace)+'.receipt.json')])
    s.run('sensitivity-partitions',[python,here/'sensitivity_data.py','--trace',trace,'--out',root/'measurement-traces',
          '--rows',str(plan['measurement_rows']),'--h-rows',str(plan['hessian_rows']),'--critical-min',str(plan['measurement_critical_min'])],
          [root/'measurement-traces/selection.json',root/'measurement-traces/A.safetensors',root/'measurement-traces/B.safetensors'])
    s.run('teacher-forward',[python,here/'check_teacher.py','--assets',assets,'--model',source,'--out',root/'teacher-forward.json'],[root/'teacher-forward.json'])
    s.run('gradient-parity',[python,here/'gpu_parity.py','--out',root/'gradient-parity.json'],[root/'gradient-parity.json'])
    s.run('new-width-runtime',[python,here/'role_probe.py','--plan',root/'plan.json'],[root/'role-probe/complete.json'])
    # A real early-layer full-backward pilot tests HF/GDN offload and resource use.
    from safetensors.numpy import load_file
    import numpy as np
    pilot_offset=int(np.argmax(load_file(str(trace))['lengths']))
    s.run('yaqa-pilot',[python,here/'yaqa.py','--model-dir',source,'--calibration',trace,'--out',root/'yaqa-pilot',
          '--layer','0','--layer-end','2','--rows','1','--skip-rows',str(pilot_offset),'--cols',str(plan['yaqa_cols'])],[root/'yaqa-pilot/collector.json'])
    s.run('two-sided-pilot',[python,here/'fit_pilot.py','--plan',root/'plan.json'],[root/'two-sided-pilot.json'])
    s.run('actual-k2-k3-rates',[python,here/'rate_probes.py','--plan',root/'plan.json'],[root/'rate-probes.json'])
    require(read(root/'rate-probes.json')['recommended_max_noise_scale']<=max(plan['rfn_scales']),
            'Actual K2/K3 rate ratio falls outside the frozen noise range; inspect and extend a new measurement plan')
    s.run('rfn',[python,TREE/'sc_rfn_probe.py','-mq',v1/'full/converted','-mr',source,'-d','0','-o',root/'rfn.json'],[root/'rfn.json'])
    anchor={r['key']:r['rfn'] for r in read(root/'rfn.json')['results']}
    measured=read(root/'rate-probes.json')['results'];coverage={}
    for value in measured:
        key=value['identity']['key'];coverage[key]=max(coverage.get(key,0),value['rfn']/anchor[key])
    atomic_json(root/'rate-coverage.json',dict(actual_YaQA_over_V1_anchor=coverage,measured_noise_max=max(plan['rfn_scales']),
        scope='Representative roles only; all other tensors retain a modeled rate curve'))
    require(all(v<=max(plan['rfn_scales']) for v in coverage.values()),'Actual rate probes exceed the measured V1-anchor noise envelope')
    for replicate in plan['measurement_replicates']:
        s.run('shaped-sensitivity-'+replicate,[python,here/'measure.py','--contract',root/'contract.json','-m',source,'--streaming','--max-sys',str(plan['measurement_memory_gb']),
          '-tr',root/f'measurement-traces/{replicate}.safetensors','-r',str(plan['measurement_rows']),'-l',str(plan['measurement_cols']),
          '--shaped','--h_rows',str(plan['hessian_rows']),'--draws',str(plan['measurement_draws']),
          '--rfn_ref',root/'rfn.json','--rfn_scale',','.join(map(str,plan['rfn_scales'])),'-o',root/f'sensitivity-{replicate}.json'],
          [root/f'sensitivity-{replicate}.json'],timeout=72*3600)
    s.run('allocation',[python,here/'allocate.py','--contract',root/'contract.json','--measurement',root/'sensitivity-A.json','--replicate',root/'sensitivity-B.json',
          '--rfn',root/'rfn.json','--out',root/'allocation'],[root/'allocation/recipe.yaml',root/'allocation/allocation-audit.json'])
    factors=root/'hessians';factors.mkdir(exist_ok=True)
    for start in range(0,64,plan['yaqa_chunk']):
        end=min(start+plan['yaqa_chunk'],64);store=Path(plan['factor_stores'][(start//plan['yaqa_chunk'])%len(plan['factor_stores'])]);out=store/f'{start:02d}-{end:02d}'
        s.run(f'yaqa-{start:02d}-{end:02d}',[python,here/'yaqa.py','--model-dir',source,'--calibration',trace,'--out',out,
              '--layer',str(start),'--layer-end',str(end),'--rows',str(plan['yaqa_rows']),'--cols',str(plan['yaqa_cols']),
              '--sides',plan['yaqa_sides'],'--samples',str(plan['yaqa_samples'])],[out/'collector.json'],timeout=48*3600)
        receipt=read(out/'collector.json')
        expected={r['source_key']+'.safetensors' for r in policy['projections'] if start<=r['layer']<end}
        require(set(receipt['files'])==expected,'YaQA range failed to cover every projection')
        for name,h in receipt['files'].items():
            require(sha(out/name)==h,'Factor changed before publication')
            if (factors/name).exists():require(sha(factors/name)==h,'Conflicting factor')
            elif (factors/name).is_symlink():require((factors/name).resolve()==(out/name).resolve(),'Conflicting factor link')
            else:(factors/name).symlink_to(out/name)
    require(len(list(factors.glob('*.safetensors')))==400,'Missing two-sided factors')
    s.run('native-ablation-and-selection',[python,here/'candidates.py','--plan',root/'plan.json'],[root/'selection.complete.json'],timeout=7*24*3600)
    atomic_json(root/'complete.json',dict(pipeline_complete=True,qualified=read(root/'selection.complete.json'),plan_sha256=sha(root/'plan.json'),promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--work',required=True);prep.add_argument('--assets',required=True)
    prep.add_argument('--v1',required=True);prep.add_argument('--template',required=True);prep.add_argument('--bridge-source',required=True)
    prep.add_argument('--trace-model',help='Optional already available high-precision EXL3 teacher for native sc_trace')
    execute=sub.add_parser('run');execute.add_argument('--work',required=True)
    a=p.parse_args();prepare(a) if a.action=='prepare' else run(a)
