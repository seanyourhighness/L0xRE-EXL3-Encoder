"""Native ablations isolate fitting, allocation and their combination before selection."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import math
import numpy as np
from common import TREE,read,require,sha,atomic_json,validate_recipe,native_runtime


def native_gate(report,require_critical=True):
    reasons=[]
    core=[report['candidate_mean'],report['baseline_mean'],*report['paired_document95']]
    if not all(math.isfinite(v) for v in core):reasons.append('nonfinite native quality statistics')
    if require_critical and not {'code','tools','reasoning'}<=set(report.get('domains',{})):reasons.append('missing critical development domains')
    if not {'p99','p99_9'}<=set(report.get('tails',{})):reasons.append('missing native tail evidence')
    if report['candidate_mean']>=report['baseline_mean']:reasons.append('mean KLD not better than B02-A2')
    if report['paired_document95'][0]<=0:reasons.append('mean improvement not resolved by paired document95CI')
    for key,value in report.get('domains',{}).items():
        if not value['within_margin_before_AA_jitter']:reasons.append('domain margin: '+key)
        if key in ('code','tools','reasoning') and value['candidate_mean']>value['baseline_mean']:reasons.append('critical domain mean: '+key)
    for key,value in report.get('tails',{}).items():
        if not value['within_margin_before_AA_jitter']:reasons.append('tail margin: '+key)
    return dict(passed=not reasons,reasons=reasons,scope='Paired native development selection, not final release qualification')


def cleanup(candidate):
    """Only remove reproducible scratch after verified final GGUF + native output."""
    candidate=Path(candidate);receipt=read(candidate/'native.complete.json')
    require(sha(candidate/'quant.gguf')==receipt['candidate_sha256'],'Cannot reclaim scratch before verified native candidate')
    if (candidate/'scratch-cleanup.json').exists():return
    removed=[]
    for name in ('conversion/qtensors','conversion/ckpt','conversion/ckpt_old','converted','export-checkpoints'):
        path=candidate/name
        if path.exists():
            require(not path.is_symlink() and path.resolve().is_relative_to(candidate.resolve()),'Unsafe scratch cleanup path')
            count=sum(p.stat().st_size for p in path.glob('**/*') if p.is_file());shutil.rmtree(path);removed.append(dict(path=name,bytes=count))
    atomic_json(candidate/'scratch-cleanup.json',dict(candidate_sha256=receipt['candidate_sha256'],removed=removed,
       reason='Preserve all final model weights/native rows/source pins/factors; reclaim only owned reproducible fit/export scratch to keep the remaining ablations within SSD budget'))


def compiled_checkpoint(candidate,policy,recipe):
    """The durable compiled shards replace temporary per-module/state copies."""
    from safetensors import safe_open
    candidate=Path(candidate);out=candidate/'compiled-checkpoint.json'
    if out.exists():
        for p,h in read(out)['shards'].items():require(sha(candidate/p)==h,'Compiled checkpoint changed')
        return
    keys={};shards=sorted((candidate/'converted').glob('*.safetensors'));require(bool(shards),'No compiled fit checkpoint')
    for file in shards:
        with safe_open(file,framework='np') as f:
            for key in f.keys():require(key not in keys,'Duplicate compiled tensor');keys[key]=file
    for r in policy['projections']:
        key=r['source_key'];require(all(key+'.'+s in keys for s in ('trellis','suh','svh','mcg')),'Incomplete compiled model')
        with safe_open(keys[key+'.trellis'],framework='np') as f:
            require(tuple(f.get_slice(key+'.trellis').get_shape())==(r['ic']//16,r['oc']//16,16*recipe['tensors'][key]),'Compiled width differs')
    proof=dict(shards={str(f.relative_to(candidate)):sha(f) for f in shards},reason='Durable compiled tensor checkpoint retains every fitted weight before removing redundant module/state scratch')
    atomic_json(out,proof)
    for name in ('conversion/qtensors','conversion/ckpt','conversion/ckpt_old'):
        p=candidate/name
        if p.exists():require(not p.is_symlink() and p.resolve().is_relative_to(candidate.resolve()),'Unsafe temporary path');shutil.rmtree(p)


def run(a):
    from run import Stages
    plan=read(a.plan);root=Path(plan['work']);s=Stages(root);here=Path(__file__).parent;python=sys.executable
    trace=root/'calibration.safetensors';factors=root/'hessians';policy=plan['policy'];allocation=read(root/'allocation/recipe.yaml')
    changed=bool(validate_recipe(policy,allocation)['changed_projections'])
    specs=[('fixed-one-sided',root/'allocation/fixed-recipe.yaml',False),('fixed-yaqa',root/'allocation/fixed-recipe.yaml',True)]
    if changed:specs.extend([('allocation-only',root/'allocation/recipe.yaml',False),('combined',root/'allocation/recipe.yaml',True)])
    records=[]
    # The head-only carry-forward is a measured development candidate too; do
    # not discard it if expensive fits fail to beat an already-earned head.
    control=read(root/'head-control-native.complete.json');head=root/'candidates/V1-body-production-head';head.mkdir(parents=True,exist_ok=True)
    child=dict(plan,work=str(head),candidate_label='V1-body-production-head',parent_plan=str(root/'plan.json'),fit_method='No new fit; existing V1 body and unchanged production vocabulary')
    atomic_json(head/'plan.json',child)
    for link,target in [(head/'quant.gguf',Path(plan['head_control_model'])),
                        (head/'quant.gguf.receipt.json',Path(str(plan['head_control_model'])+'.receipt.json')),
                        (head/'role-probe',root/'role-probe')]:
        if not link.exists():link.symlink_to(target)
    require(sha(head/'quant.gguf')==control['candidate_sha256'],'Head control model changed')
    require(sha(root/'head-control-native.json')==control['report_sha256'] and sha(str(root/'head-control-native.json')+'.rows')==control['rows_sha256'],
            'Head control native rows changed')
    for target,source in [('development116.json',root/'head-control-native.json'),('development116.json.rows',Path(str(root/'head-control-native.json')+'.rows')),
                          ('development116.log',root/'head-control-native.log')]:
        p=head/target
        if not p.exists():shutil.copy2(source,p)
    s.run('head-control-paired',[python,here/'qualify.py','--plan',head/'plan.json','--native-only'],[head/'native.complete.json'])
    report=read(head/'paired-B02-A2.json');records.append(dict(label='V1-body-production-head',model=str(head/'quant.gguf'),
        sha256=control['candidate_sha256'],mean=report['candidate_mean'],gate=native_gate(report),native_report_sha256=sha(head/'paired-B02-A2.json')))
    for label,recipe,yaqa in specs:
        candidate=root/'candidates'/label;candidate.mkdir(parents=True,exist_ok=True)
        child=dict(plan,work=str(candidate),candidate_label=label,parent_plan=str(root/'plan.json'),fit_method='two-sided-YaQA' if yaqa else 'V1-one-sided')
        path=candidate/'plan.json'
        if path.exists():require(read(path)==child,'Candidate plan changed')
        else:atomic_json(path,child)
        link=candidate/'role-probe'
        if not link.exists():link.symlink_to(root/'role-probe',target_is_directory=True)
        if not (candidate/'native.complete.json').exists():
            atomic_json(candidate/'fit-context.json',dict(recipe_sha256=sha(recipe),calibration_sha256=sha(trace),
                factors={f.name:sha(f) for f in sorted(factors.glob('*.safetensors'))} if yaqa else {},
                fit_method=child['fit_method'],source_revision=plan['source_revision'],codebook='mcg'))
            cmd=[python,here/'resume_convert.py','--v2-context',candidate/'fit-context.json','-i',plan['source'],
                '-w',candidate/'conversion','-o',candidate/'converted','-rcp',recipe,'-cb','mcg','-hb','16','-eb','16','-vb','16','-mb','16',
                '-cd',trace,'-cr',str(plan['calibration_rows']),'-cc',str(plan['calibration_cols']),'--scaled_hessian','-d','0','-cpi','120']
            if yaqa:cmd+=['--hessians',factors,'--out_scales','yaqa']
            s.run(label+'-fit',cmd,[candidate/'converted/quantization_config.json'],timeout=48*3600)
            compiled_checkpoint(candidate,policy,read(recipe))
            s.run(label+'-export',[python,here/'pack.py','--contract',root/'contract.json','--recipe',recipe,
                '--checkpoint',candidate/'converted','--work',candidate/'export-checkpoints','--output',candidate/'quant.gguf',
                '--template-sha256',plan['template_sha256'],'--workers',str(plan['pack_workers'])],
                [candidate/'quant.gguf',candidate/'quant.gguf.receipt.json'],timeout=24*3600)
            s.run(label+'-native',[python,here/'qualify.py','--plan',path,'--native-only'],[candidate/'native.complete.json'],timeout=4*3600)
        report=read(candidate/'paired-B02-A2.json');gate=native_gate(report)
        record=dict(label=label,model=str(candidate/'quant.gguf'),sha256=sha(candidate/'quant.gguf'),mean=report['candidate_mean'],gate=gate,
                    native_report_sha256=sha(candidate/'paired-B02-A2.json'))
        records.append(record);atomic_json(root/'development-candidates.json',dict(candidates=records,final_holdout_opened=False))
        cleanup(candidate)
        print('NATIVE_CANDIDATE',record,flush=True)
    eligible=[r for r in records if r['gate']['passed']]
    if not eligible:
        atomic_json(root/'selection.complete.json',dict(complete=True,selected=None,candidates=records,promoted=False,
               reason='No candidate passes paired mean/domain/tail gates vs production B02-A2; retain current production',holdout_opened=False));return
    winner=min(eligible,key=lambda r:r['mean']);candidate=Path(winner['model']).parent
    atomic_json(root/'development-winner.json',dict(winner=winner,selection_before_holdout=True,candidates=records))
    s.run('winner-coding-DFlash',[python,here/'qualify.py','--plan',candidate/'plan.json'],[candidate/'qualification.complete.json'],timeout=24*3600)
    quality=read(candidate/'qualification.complete.json')
    if not (quality['coding_qualified'] and quality['dflash_qualified']):
        atomic_json(root/'selection.complete.json',dict(complete=True,selected=winner,candidates=records,promoted=False,
              coding_qualified=quality['coding_qualified'],dflash_qualified=quality['dflash_qualified'],holdout_opened=False,
              reason='Development winner failed native task/speed qualification; no promotion'));return
    from holdout import stage
    from blind_reasoning import stage as reasoning_stage
    published=stage(plan);reasoning=reasoning_stage(plan);ref=root/'final-holdout48';ref.mkdir(exist_ok=True)
    metadata={k:v for k,v in read(published/'kld.json').items() if k not in ('docs','rows','corpus_fnv1a64')};metadata.update(docs=[],rows=0)
    selection=[]
    for folder in (published,reasoning):
        info=read(folder/'kld.json')
        require(info['n_vocab']==metadata['n_vocab'] and info['top']==metadata['top'],'Holdout reference formats differ')
        for doc in info['docs']:metadata['docs'].append(dict(doc,row0=doc['row0']+metadata['rows']))
        metadata['rows']+=info['rows'];selection.extend(read(folder/'selection.json'))
    for name in ('tokens.i32','top.bin','rows.f32','gen.u8'):
        with (ref/name).open('wb') as out:
            for folder in (published,reasoning):
                with (folder/name).open('rb') as inp:shutil.copyfileobj(inp,out,4<<20)
    atomic_json(ref/'kld.json',metadata);atomic_json(ref/'selection.json',selection)
    atomic_json(ref/'receipt.json',dict(published_sha256=sha(published/'receipt.json'),reasoning_sha256=sha(reasoning/'receipt.json'),
                 files={n:sha(ref/n) for n in ('tokens.i32','top.bin','rows.f32','gen.u8','kld.json','selection.json')}))
    runtime=native_runtime(plan)
    baseline=Path(plan['assets'])/'artifacts/stacking-loop/B02/A2/quant.gguf'
    require(sha(baseline)=='ecdd8679d7cfe023cadb3eb8726fb09a869a6f71f69c183d71144973b0785bc7','Production holdout baseline changed')
    for label,model in [('B02-A2',baseline),('winner',Path(winner['model']))]:
        command=[str(Path(plan['assets'])/'tools/native_kld'),'--model',str(model),'--ref',str(ref),'--out',str(root/('holdout-'+label+'.json')),
             '--ctx','8192','--batch','64','--ubatch','64','--threads','6']
        path=root/('holdout-'+label+'.json')
        if not path.exists():
            with (root/('holdout-'+label+'.log')).open('wb') as log:
                subprocess.run(command,env=dict(os.environ,**runtime['environment']),stdout=log,stderr=subprocess.STDOUT,check=True)
        import re
        counts=re.findall(r'L0XRE_FASTPATH_SUMMARY fallthroughs=(\d+)',(root/('holdout-'+label+'.log')).read_text(errors='replace'))
        values=np.fromfile(str(path)+'.rows',np.float32)
        require(counts and all(int(v)==0 for v in counts) and len(values)>0 and np.isfinite(values).all(),'Holdout native execution failed')
    paired=root/'paired-holdout48.json'
    subprocess.run([python,str(Path(plan['assets'])/'tools/paired_quality_subset.py'),'--baseline',str(root/'holdout-B02-A2.json'),
       '--candidate',str(root/'holdout-winner.json'),'--ref',str(ref),'--candidate-sha256',winner['sha256'],'--out',str(paired)],check=True)
    final=native_gate(read(paired))
    coverage=read(root/'holdout-reservation.json')['critical_domain_coverage']
    coverage['reasoning']=len(read(root/'blind-reasoning-reservation.json')['cases'])
    atomic_json(root/'selection.complete.json',dict(complete=True,selected=winner,candidates=records,promoted=False,
       coding_qualified=True,dflash_qualified=True,holdout_opened=True,holdout_gate=final,native_validation_complete=True,
       holdout_domain_coverage=coverage,eligible_for_separate_promotion_review=bool(final['passed'] and all(coverage.values())),
       promotion_requires_owner_go_ahead=True,coverage_limitation='Reasoning final holdout uses16new frozen constructed arithmetic tasks; published32 exclude known historical exposures, not provable all historical exposure'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);run(p.parse_args())
